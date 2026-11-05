"""v2: the interior-byte decoder — the missing half, trained against the frozen trunk.

    colab run --gpu A100 kernel/train_decoder.py

Pipeline per doc: periodic-6 spans -> patch sums -> frozen trunk -> hidden state at position j-1
(the context that predicts chunk j) -> ByteDecoder predicts the chunk's bytes teacher-forced.
Loss: byte CE on interior positions.  Reports the REAL full-byte bpb: boundary bytes by the trunk
head, interior bytes by the decoder, /T — the first honest LM-number of the composed system.
TRAIN_PROJ=1 also trains the rank-32 patch map first (the v1 recipe) before the decoder.
"""
import os, subprocess, sys, time

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


print("[setup] cloning", flush=True)
for url, path in (("https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO),
                  ("https://github.com/KidIkaros/hnet-kaggle.git", "/content/qt/hnet-kaggle")):
    if os.path.exists(path):
        sh("git", "-C", path, "pull", "-q")
    else:
        sh("git", "clone", "-q", url, path)
sys.path.insert(0, REPO)
os.chdir(REPO)

print("[setup] model", flush=True)
import torch                                                     # noqa: E402
import numpy as np                                               # noqa: E402
import torch.nn.functional as F                                  # noqa: E402
from huggingface_hub import snapshot_download                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
src = snapshot_download("Qwen/Qwen3.5-0.8B")

from src.adapter import ByteHead, PatchEmbedder, PeriodicChunker  # noqa: E402
from src.corpus import byte_token_map                             # noqa: E402
from src.decoder import ByteDecoder, chunk_batch, SOS             # noqa: E402
from src.learned import PatchProjection, byte_logprobs            # noqa: E402

tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
pe = PatchEmbedder(model, tok)
head = ByteHead(tok)

# ---- corpus
buf = []
for name in sorted(os.listdir("/content/qt/hnet-kaggle/docs")):
    if name.endswith(".md"):
        buf.append(open(os.path.join("/content/qt/hnet-kaggle/docs", name), "rb").read())
joined = b"\n".join(buf)
NDOCS = int(os.environ.get("NDOCS", "16"))
docs = [joined[i:i + 8192] for i in range(0, len(joined) - 8192, 8192)][:NDOCS]
train_docs, eval_docs = docs[:-3], docs[-3:]
print(f"[corpus] {len(train_docs)} train + {len(eval_docs)} eval", flush=True)

P = 6
chunker = PeriodicChunker(P)
emb = pe.emb.detach().float().cuda()
PAD = emb.shape[0]
emb_ext = torch.cat([emb, torch.zeros(1, emb.shape[1], device="cuda")])
D = emb.shape[1]
proj = PatchProjection(D, rank=32).cuda()


def spans_of(data):
    sp = chunker.spans(data)
    tail = len(data) - sp[-1][0] - P if sp else len(data)
    if len(data) % P:
        sp = sp + [(len(data) - len(data) % P, data[len(data) - len(data) % P:])]
    return sp


def base_sums(spans):
    ids = torch.full((len(spans), P), PAD, dtype=torch.long)
    for m, (_, c) in enumerate(spans):
        for i, b in enumerate(c):
            if b in bmap:
                ids[m, i] = bmap[b]
    return emb_ext[ids.cuda()].sum(1)


def trunk_ctx(patch):
    """(M, D) trunk last-hidden per chunk position; ctx for chunk j = hidden[j-1] (j=0 -> ctx0)."""
    with torch.no_grad():
        out = model(inputs_embeds=patch.unsqueeze(0).to(torch.bfloat16),
                    output_hidden_states=True)
        hs = out.hidden_states[-1][0].float()                   # (M, D)
        lg = out.logits[0]                                      # (M, V)
    ctx = torch.cat([torch.zeros(1, D, device=hs.device), hs[:-1]])
    return ctx, hs, lg


# ---- optional: train the patch map first (the v1 recipe, inline)
if os.environ.get("TRAIN_PROJ", "0") == "1":
    opt = torch.optim.AdamW(proj.parameters(), lr=3e-3)
    for ep in range(6):
        for d in train_docs:
            sp = spans_of(d)
            base = base_sums(sp)
            patch = proj(base)
            ctx, hs, lg = trunk_ctx(patch)
            lp = byte_logprobs(lg[:-1], head.first)             # pos j-1 predicts chunk j's byte0
            tgt = torch.tensor([sp[j + 1][1][0] for j in range(len(sp) - 1)], device="cuda")
            loss = -lp[torch.arange(len(tgt)), tgt].mean()
            opt.zero_grad(); loss.backward(); opt.step()
    for p in proj.parameters():
        p.requires_grad_(False)
    print("[v2] patch projection trained inline (6 epochs)", flush=True)

# ---- decoder training
dec = ByteDecoder(D, max_len=P, d_dec=256, n_layers=4, n_heads=4).cuda()
opt = torch.optim.AdamW(dec.parameters(), lr=float(os.environ.get("LR", "1e-3")))
EPOCHS = int(os.environ.get("EPOCHS", "8"))

t0 = time.time(); step = 0
for ep in range(EPOCHS):
    ep_tot = ep_n = 0.0
    for i in np.random.RandomState(ep).permutation(len(train_docs)):
        d = train_docs[i]
        sp = spans_of(d)
        base = base_sums(sp)
        ctx, hs, lg = trunk_ctx(proj(base))
        x, y, pad = [t.cuda() for t in chunk_batch(d, sp, max_len=P)]
        logits = dec(ctx, x, pad)
        interior = ~pad.clone(); interior[:, 0] = False       # p>=1: interior bytes only
        loss = F.cross_entropy(logits[interior].float(), y[interior])
        opt.zero_grad(); loss.backward(); opt.step()
        ep_tot += float(loss) * int(interior.sum()); ep_n += int(interior.sum()); step += 1
    print(f"[epoch {ep:02d}] interior-byte train bpb {ep_tot / ep_n / np.log(2):.4f} ({step} steps)",
          flush=True)
print(f"[timing] decoder training {time.time() - t0:.0f}s", flush=True)

# ---- the real full-byte eval
tot_bits = tot_bytes = 0.0
for d in eval_docs:
    sp = spans_of(d)
    base = base_sums(sp)
    ctx, hs, lg = trunk_ctx(proj(base))
    bits = 8.0                                                 # byte 0: uniform
    lp = byte_logprobs(lg[:-1], head.first)                    # boundary bytes by the trunk head
    for j in range(len(sp) - 1):
        bits += float(-lp[j, sp[j + 1][1][0]])
    x, y, pad = [t.cuda() for t in chunk_batch(d, sp, max_len=P)]
    with torch.no_grad():
        logits = dec(ctx, x, pad)
    lsm = logits.float().log_softmax(-1)
    interior = ~pad.clone(); interior[:, 0] = False
    bits += float(-lsm[interior, y[interior]].sum())
    tot_bits += bits; tot_bytes += len(d)

print(f"\n[v2] FULL-BYTE bpb on eval docs: {tot_bits / np.log(2) / tot_bytes:.4f} "
      f"(boundary=trunk, interior=decoder)", flush=True)
print("[refs] byte-level 5.57 | constructed-sum eff 5.56 | v1 eff 5.27 | stock BPE 4.92 "
      "(eval docs)", flush=True)
torch.save(dec.state_dict(), "/content/byte_decoder.pt")
print("[save] /content/byte_decoder.pt", flush=True)
