"""Closed-form layer-boundary adapters for the re-sliced H-Net-of-Qwen.

    colab run --gpu A100 kernel/hnet_adapt.py

Harvests paired activations by running the FULL trunk at byte level:
  w_in  : byte-emb        -> input to the first decoder slice layer
  w_out : norm(dec_out)   -> final-layer hidden state (lm_head input)
Fit via the same ridge recipe as the patch map.  Evals byte bpb with/without.
"""
import os, subprocess, sys

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

src = snapshot_download("Qwen/Qwen3.5-0.8B")
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
hn = HNetQwen(model, tok, bmap, dec_layers=14).cuda()
hn.dechunk.weight.data.zero_()
head = ByteHead(tok)

docs = [open(f, "rb").read() for f in sorted(glob.glob("data/decontam/*.bin"))[:8]]
print(f"[adapt] {len(docs)} docs", flush=True)

# ---- harvest: full-trunk hidden states at byte positions
# the decoder slice = the last 14 DeltaNet layers -> find its first layer index
lin_idx = [i for i, l in enumerate(hn.layers) if l.block_type == "linear_attention"]
dec_start = lin_idx[-14]                                        # first decoder slice
text = model.model if hasattr(model, "model") else model.language_model
if not hasattr(text, "layers"):
    text = text.language_model

attn_idx = [i for i, l in enumerate(hn.layers) if l.block_type == "full_attention"]
main_start = attn_idx[0]                                       # first attn layer
Xin, Yin, Xout, Yout, Ymain = [], [], [], [], []
for d in docs:
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    with torch.no_grad():
        o = text(input_ids=tids[None], output_hidden_states=True)
    hs_layers = o.hidden_states                                # (L+1, 1, T, D)
    emb_in = text.embed_tokens(tids).float()                   # byte embs (T, D)
    Xin.append(emb_in)
    Yin.append(hs_layers[dec_start][0].float())                # dec-slice input
    Yout.append(text.norm(hs_layers[-1][0]).float())
    Ymain.append(hs_layers[main_start][0].float())             # attn-layer input
    del o, hs_layers
    torch.cuda.empty_cache()

Xin, Yin, Yout, Ymain = (torch.cat(Xin), torch.cat(Yin),
                         torch.cat(Yout), torch.cat(Ymain))
# decoder output under the re-sliced path: run dec on REAL dec inputs (paired)
dec_out = []
for d in docs[:4]:
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

# encoder outputs at chunk-boundary (last-byte) positions -> attn input target
P = 6
Xmain = []
for d in docs[:4]:
    tids = torch.tensor([bmap.get(int(b), 0) for b in d], device="cuda")
    hs = text.embed_tokens(tids[None])
    for layer in hn.enc:
        hs = layer(hs, position_embeddings=None, attention_mask=None,
                   position_ids=None)
    enc = hs[0]
    bounds = [b - 1 for b in range(P, len(d) + 1, P)]
    Xmain.append(enc[torch.tensor(bounds, device="cuda")].float())
Xmain = torch.cat(Xmain)
Ymain_p = []
off = 0
for d in docs[:4]:
    n = len(d) // P
    Ymain_p.append(Ymain[off + P - 1:off + n * P:P])
    off += len(d)
Ymain_p = torch.cat(Ymain_p)

print(f"[adapt] pairs {len(Xin)} | dec_start {dec_start} main_start {main_start} "
      f"| w_main pairs {len(Xmain)}", flush=True)
hn.w_in = ridge_to_bpe(Xin[:200_000].to(torch.bfloat16), Yin[:200_000].to(torch.bfloat16))
hn.w_out = ridge_to_bpe(Xout[:200_000].to(torch.bfloat16), Yout[:len(Xout)].to(torch.bfloat16))
hn.w_main = ridge_to_bpe(Xmain[:200_000].to(torch.bfloat16), Ymain_p[:200_000].to(torch.bfloat16))

# ---- eval grid: adapters on/off x ctx prev/retrieve
T = 8192
byte_ids = torch.tensor(list(docs[0]), device="cuda")
b6 = torch.zeros(T, device="cuda"); b6[::6] = 1.0
from src.learned import byte_logprobs                          # noqa: E402


def eval_bpb(w_main, ctx):
    hn.w_main = w_main
    with torch.no_grad():
        bp, _, _ = hn(byte_ids, boundaries=b6, ctx=ctx,
                      head_first=head.first)
    nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    return float(nll.sum()) / np.log(2) / (T - 1)


for wm, tag_m in ((None, "no w_main"), (hn.w_main, "+w_main")):
    for ctx, tag_c in (("prev", "prev"), ("retrieve", "retrieve")):
        print(f"[adapt] {tag_m:<10} | {tag_c:<8} byte bpb {eval_bpb(wm, ctx):.4f}",
              flush=True)
