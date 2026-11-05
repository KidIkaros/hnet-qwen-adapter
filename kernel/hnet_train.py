"""DRY-RUN trainer: end-to-end byte-NLL fine-tune of the re-sliced H-Net-of-Qwen.

    colab run --gpu A100 kernel/hnet_train.py

2B-Base, enc=6/dec=8 DeltaNet slices + dechunk + boundary UNFROZEN;
emb/main/lm_head frozen.  Ridge adapters as frozen init.  Periodic-6
boundaries this pass (boundary-head training needs H-Net's smoothing aux —
next iteration).  ~60 docs, few steps — smoke test: does the loop run,
does loss drop, does it fit?
"""
import os, subprocess, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q", "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch                                                     # noqa: E402
import numpy as np                                               # noqa: E402
import glob                                                      # noqa: E402
from huggingface_hub import snapshot_download                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
from src.corpus import byte_token_map                             # noqa: E402
from src.adapter import ByteHead                                  # noqa: E402
from src.hnet import HNetQwen                                     # noqa: E402
from src.construct import ridge_to_bpe                            # noqa: E402
from src.learned import byte_logprobs                             # noqa: E402

MODEL = "Qwen/Qwen3.5-2B-Base"
src = snapshot_download(MODEL)
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda()
for p in model.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
head = ByteHead(tok)
text = model.model if hasattr(model, "model") else model.language_model
if not hasattr(text, "layers"):
    text = text.language_model
P = 6
ENC, DEC = 6, 8
hn = HNetQwen(model, tok, bmap, enc_layers=ENC, dec_layers=DEC).cuda()
lin_idx = [i for i, l in enumerate(hn.layers) if l.block_type == "linear_attention"]
dec_start = lin_idx[-DEC]
print(f"[train] {MODEL} | {len(hn.layers)} layers | "
      f"DeltaNet {len(lin_idx)} | enc {ENC} dec {DEC} (start {dec_start})",
      flush=True)

# ---- data: decontam docs for the dry run (no external download needed)
all_docs = [open(f, "rb").read() for f in sorted(glob.glob("data/decontam/*.bin"))]
fit_docs = [d[:4096] for d in all_docs[:4]]
train_docs = [d[:4096] for d in all_docs[4:38]]
held_docs = [d[:4096] for d in all_docs[38:44]]
print(f"[data] {len(train_docs)} train + {len(held_docs)} held + "
      f"{len(fit_docs)} adapter-fit docs", flush=True)


# ---- adapter fit (frozen init): w_in/w_out/w_enc
def harvest(d):
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    with torch.no_grad():
        o = text(input_ids=tids[None], output_hidden_states=True)
    return text.embed_tokens(tids), o.hidden_states


Xin, Yin, Yout, Xenc = [], [], [], []
for d in fit_docs:
    emb, hsl = harvest(d)
    Xin.append(emb.float()); Yin.append(hsl[dec_start][0].float())
    Yout.append(text.norm(hsl[-1][0]).float())
    hs = emb[None]
    with torch.no_grad():
        for layer in hn.enc:
            hs = layer(hs, position_embeddings=None, attention_mask=None,
                       position_ids=None)
    Xenc.append(hs[0].float())
    del hsl, hs
    torch.cuda.empty_cache()
Xin, Yin, Yout, Xenc = (torch.cat(x) for x in (Xin, Yin, Yout, Xenc))
hn.w_in = ridge_to_bpe(Xin[:200_000].to(torch.bfloat16),
                       Yin[:200_000].to(torch.bfloat16))
dec_out = []
for d in fit_docs:
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    with torch.no_grad():
        o = text(input_ids=tids[None], output_hidden_states=True)
    hs = o.hidden_states[dec_start][0][None].to(torch.bfloat16)
    for layer in hn.dec:
        hs = layer(hs, position_embeddings=None, attention_mask=None,
                   position_ids=None)
    dec_out.append(text.norm(hs[0]).float())
    del o
    torch.cuda.empty_cache()
Xout = torch.cat(dec_out)
hn.w_out = ridge_to_bpe(Xout[:200_000].to(torch.bfloat16),
                        Yout[:len(Xout)].to(torch.bfloat16))
enc_resid = (Yin[:len(Xenc)] - Xin[:len(Xenc)]).to(torch.bfloat16)
hn.w_enc = ridge_to_bpe(Xenc[:200_000].to(torch.bfloat16), enc_resid[:200_000])
print("[fit] w_in/w_out/w_enc ridges fit (frozen during training)", flush=True)

# ---- trainable: enc, dec, dechunk, boundary — with grad checkpointing
for layer in list(hn.enc) + list(hn.dec):
    layer.gradient_checkpointing_enable()
trainable = list(hn.enc.parameters()) + list(hn.dec.parameters()) + \
    list(hn.dechunk.parameters()) + list(hn.boundary.parameters())
for p in trainable:
    p.requires_grad_(True)
n_train = sum(p.numel() for p in trainable)
print(f"[train] {n_train/1e6:.1f}M trainable params (rest frozen)", flush=True)
opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.01)


def doc_loss(byte_ids):
    T = len(byte_ids)
    b6 = torch.zeros(T, device="cuda"); b6[::P] = 1.0
    bp, _, _ = hn(byte_ids, boundaries=b6, ctx="retrieve", n_ctx=1,
                  head_first=head.first)
    nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    return nll.mean()


def eval_bpb(byte_ids):
    with torch.no_grad():
        return float(doc_loss(byte_ids)) / np.log(2)


for step, d in enumerate(train_docs):
    byte_ids = torch.tensor(list(d), device="cuda")
    loss = doc_loss(byte_ids)
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    if step % 10 == 0 or step == len(train_docs) - 1:
        hb = eval_bpb(torch.tensor(list(held_docs[step % len(held_docs)]),
                                 device="cuda"))
        print(f"  step {step:<4} loss {float(loss):.4f} |gn| {float(gn):.2f} "
              f"| held bpb {hb:.4f}", flush=True)
    del byte_ids, loss
    torch.cuda.empty_cache()
print("[train] dry-run complete", flush=True)
