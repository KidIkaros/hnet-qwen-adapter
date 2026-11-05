"""Conjugated transfer: does the learned patch-map direction transfer across sizes?

    colab run --gpu A100 kernel/conjugate_transfer.py

dW_t = R @ dW_s @ R+  — the learned 0.8B residual conjugated through a closed-form ridge
bridge R fit on the ~150k paired shared-vocab embedding rows (0.8B -> 4B-Base).  If the
transferred map beats identity on 4B-Base, the learned direction is size-invariant.
Controls: identity and a locally-refit ridge->BPE map (the construction that already works).
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
import gc                                                        # noqa: E402
from huggingface_hub import snapshot_download                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
from src.adapter import ByteHead, PatchEmbedder, PeriodicChunker  # noqa: E402
from src.corpus import byte_token_map                             # noqa: E402
from src.construct import ridge_to_bpe                            # noqa: E402
from src.learned import PatchProjection, byte_logprobs            # noqa: E402

buf = []
for name in sorted(os.listdir("/content/qt/hnet-kaggle/docs")):
    if name.endswith(".md"):
        buf.append(open(os.path.join("/content/qt/hnet-kaggle/docs", name), "rb").read())
joined = b"\n".join(buf)
docs = [joined[i:i + 8192] for i in range(0, len(joined) - 8192, 8192)][:16]
train_docs, eval_docs = docs[:-3], docs[-3:]
P = 6
chunker = PeriodicChunker(P)

# ---------- 1) train the rank-32 residual on 0.8B (the v1 recipe)
print("[1] train dW on 0.8B", flush=True)
s1 = snapshot_download("Qwen/Qwen3.5-0.8B")
tok = AutoTokenizer.from_pretrained(s1)
m1 = AutoModelForCausalLM.from_pretrained(s1, dtype=torch.bfloat16).cuda().eval()
for p in m1.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
emb1 = PatchEmbedder(m1, tok).emb.detach().float().cuda()
PAD = emb1.shape[0]
ext1 = torch.cat([emb1, torch.zeros(1, emb1.shape[1], device="cuda")])
head = ByteHead(tok)


def prep(data, ext, P_):
    spans = chunker.spans(data)
    ids = torch.full((len(spans), P_), PAD, dtype=torch.long)
    for m, (_, c) in enumerate(spans):
        for i, b in enumerate(c[:P_]):
            if b in bmap:
                ids[m, i] = bmap[b]
    base = ext[ids.cuda()].sum(1)
    ms, bs = [], []
    for m in range(len(spans) - 1):
        b = int(spans[m + 1][1][0])
        if b in bmap:
            ms.append(m); bs.append(b)
    return base, torch.tensor(ms, device="cuda"), torch.tensor(bs, device="cuda")


tr1 = [prep(d, ext1, P) for d in train_docs]
D1 = emb1.shape[1]
proj = PatchProjection(D1, rank=32).cuda()
opt = torch.optim.AdamW(proj.parameters(), lr=3e-3)
for ep in range(8):
    for base, ms, bs in [tr1[i] for i in np.random.RandomState(ep).permutation(len(tr1))]:
        lg = m1(inputs_embeds=proj(base).to(torch.bfloat16).unsqueeze(0)).logits[0].float()
        lp = byte_logprobs(lg, head.first)
        loss = -lp[ms, bs].mean()
        opt.zero_grad(); loss.backward(); opt.step()
dW = (proj.A @ proj.B).detach().float()                        # (D1, D1) learned residual
print(f"    ||dW||_F {dW.norm():.2f}", flush=True)
del m1
gc.collect(); torch.cuda.empty_cache()

# ---------- 2) the ridge bridge R: emb_0.8 -> emb_4B on shared vocab
print("[2] ridge bridge R on shared vocab", flush=True)
s2 = snapshot_download("Qwen/Qwen3.5-4B-Base")
m2 = AutoModelForCausalLM.from_pretrained(s2, dtype=torch.bfloat16).cuda().eval()
for p in m2.parameters():
    p.requires_grad_(False)
emb2 = PatchEmbedder(m2, tok).emb.detach().float().cuda()
PAD2 = emb2.shape[0]
ext2 = torch.cat([emb2, torch.zeros(1, emb2.shape[1], device="cuda")])
D2 = emb2.shape[1]
n_shared = min(emb1.shape[0], emb2.shape[0])                   # same tokenizer -> row-aligned
lam = 1e-2
X, Y = emb1[:n_shared].double(), emb2[:n_shared].double()
R = torch.linalg.solve(X.T @ X + lam * torch.eye(D1, dtype=X.dtype, device="cuda"),
                       X.T @ Y).float()                       # (D1, D2)
Rp = torch.linalg.pinv(R)                                     # (D2, D1)
# row-convention: patch2 = b @ (I + conj), conj = R+ @ dW.T @ R   (D2 x D2)
conj = (Rp @ dW.T @ R).float()
print(f"    R: {n_shared} pairs, ||R+ dW.T R||_F {conj.norm():.2f}", flush=True)

# ---------- 3) eval on 4B-Base: identity vs conjugated vs local ridge
tr2 = [prep(d, ext2, P) for d in train_docs]
Xs, Ys = [], []
for base, _, _ in tr2:
    Xs.append(base)
for d in train_docs:
    for _, c in chunker.spans(d):
        tids = tok.encode(c.decode("latin-1"), add_special_tokens=False) or [0]
        Ys.append(emb2[torch.tensor(tids, device="cuda")].mean(0))
local_ridge = ridge_to_bpe(torch.cat(Xs), torch.stack(Ys))
ev2 = [prep(d, ext2, P) for d in eval_docs]


def boundary(apply):
    tot, n = 0.0, 0
    with torch.no_grad():
        for base, ms, bs in ev2:
            lg = m2(inputs_embeds=apply(base).to(torch.bfloat16).unsqueeze(0)
                    ).logits[0].float()
            lp = byte_logprobs(lg, head.first)
            tot += float(-lp[ms, bs].sum()); n += len(bs)
    return tot / n


I2 = torch.eye(D2, device="cuda")
maps = {
    "identity": lambda b: b,
    "conjugated_dW": lambda b: b + (b @ conj),
    "local_ridge": local_ridge,
}
for name, f in maps.items():
    print(f"{name:>16}  4B-Base boundary bpb {boundary(f):.4f}", flush=True)
