"""The dW lens-autopsy: what did the learned rank-32 residual actually implement?

    colab run --gpu A100 kernel/lens_autopsy.py

Retrains the v1 proj on 0.8B, SVD-canonicalizes dW (GL(r) fix), then reads the
top singular directions THROUGH the embedding table — S-lens style: right-singular
vectors -> which tokens the map reads, left-singular -> what it writes.
"""
import os, subprocess, sys

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


for url, path in (("https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO),
                  ("https://github.com/KidIkaros/hnet-kaggle.git", "/content/qt/hnet-kaggle")):
    if os.path.exists(path):
        sh("git", "-C", path, "pull", "-q")
    else:
        sh("git", "clone", "-q", url, path)
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch                                                     # noqa: E402
import numpy as np                                               # noqa: E402
from huggingface_hub import snapshot_download                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
from src.adapter import ByteHead, PatchEmbedder, PeriodicChunker  # noqa: E402
from src.corpus import byte_token_map                             # noqa: E402
from src.learned import PatchProjection, byte_logprobs            # noqa: E402

buf = []
for name in sorted(os.listdir("/content/qt/hnet-kaggle/docs")):
    if name.endswith(".md"):
        buf.append(open(os.path.join("/content/qt/hnet-kaggle/docs", name), "rb").read())
joined = b"\n".join(buf)
docs = [joined[i:i + 8192] for i in range(0, len(joined) - 8192, 8192)][:13]
P = 6
chunker = PeriodicChunker(P)

src = snapshot_download("Qwen/Qwen3.5-0.8B")
tok = AutoTokenizer.from_pretrained(src)
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in m.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
emb = PatchEmbedder(m, tok).emb.detach().float().cuda()
PAD = emb.shape[0]
ext = torch.cat([emb, torch.zeros(1, emb.shape[1], device="cuda")])
head = ByteHead(tok)

tr = []
for d in docs:
    spans = chunker.spans(d)
    ids = torch.full((len(spans), P), PAD, dtype=torch.long)
    for mm, (_, c) in enumerate(spans):
        for i, b in enumerate(c[:P]):
            if b in bmap:
                ids[mm, i] = bmap[b]
    base = ext[ids.cuda()].sum(1)
    ms, bs = [], []
    for mm in range(len(spans) - 1):
        b = int(spans[mm + 1][1][0])
        if b in bmap:
            ms.append(mm); bs.append(b)
    tr.append((base, torch.tensor(ms, device="cuda"), torch.tensor(bs, device="cuda")))

D = emb.shape[1]
proj = PatchProjection(D, rank=32).cuda()
opt = torch.optim.AdamW(proj.parameters(), lr=3e-3)
for ep in range(8):
    for base, ms, bs in [tr[i] for i in np.random.RandomState(ep).permutation(len(tr))]:
        lg = m(inputs_embeds=proj(base).to(torch.bfloat16).unsqueeze(0)).logits[0].float()
        lp = byte_logprobs(lg, head.first)
        loss = -lp[ms, bs].mean()
        opt.zero_grad(); loss.backward(); opt.step()

dW = (proj.A @ proj.B).detach().float()
U, S, Vh = torch.linalg.svd(dW)
print(f"[autopsy] ||dW||_F {dW.norm():.2f} | top sv {S[:8].tolist()}", flush=True)
print(f"[autopsy] energy: top8 {float(S[:8].sum()/S.sum()):.3f} "
      f"top16 {float(S[:16].sum()/S.sum()):.3f}", flush=True)


def read_dirs(vecs, tag, k=5):
    """Project every vocab embedding onto each direction; report top tokens."""
    for j in range(k):
        v = vecs[:, j] if vecs.dim() == 2 else vecs[j]
        sc = emb @ v                                            # (V,)
        top = torch.topk(sc, 8)
        bot = torch.topk(-sc, 8)
        dec = lambda i: repr(tok.decode([int(i)]))              # noqa: E731
        print(f"  {tag}[{j}] sv={S[j]:.2f} +{[dec(i) for i in top.indices]} "
              f"-{[dec(i) for i in bot.indices]}", flush=True)


print("[autopsy] reads (right-singular V rows: which tokens dW keys on):", flush=True)
read_dirs(Vh.mT, "V")
print("[autopsy] writes (left-singular U cols: which token-space dW writes toward):",
      flush=True)
read_dirs(U, "U")

# directional mass: does dW amplify byte-token dirs or BPE-token dirs?
byte_ids = torch.tensor(sorted(bmap.values()), device="cuda")
n_byte = emb[byte_ids].norm(dim=1).mean()
proj_byte = (emb[byte_ids] @ dW.T).norm(dim=1).mean()
oth = torch.tensor([i for i in range(emb.shape[0]) if i not in set(bmap.values())],
                   device="cuda")[:4096]
proj_oth = (emb[oth] @ dW.T).norm(dim=1).mean()
print(f"[autopsy] ||e @ dW.T|| mean: byte-tokens {proj_byte:.3f} "
      f"(||e|| {n_byte:.3f}) vs other {proj_oth:.3f} (||e|| {emb[oth].norm(dim=1).mean():.3f})",
      flush=True)
