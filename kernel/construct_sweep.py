"""The staircase experiment: hand-crafted -> closed-form -> SGD, all scored on the same eval docs.

    colab run --gpu A100 kernel/construct_sweep.py

Maps under test (all closed-form, zero gradients): identity, norm-match, covariance recolor,
paired Procrustes-ridge (patch-sum -> mean embedding of the BPE tokens covering the same bytes),
plus the rank-32 SGD projection as the trained reference.  Fits use TRAIN docs only.
"""
import os, subprocess, sys, time

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
src = snapshot_download("Qwen/Qwen3.5-0.8B")

from src.adapter import ByteHead, PatchEmbedder, PeriodicChunker, SimChunker  # noqa: E402
from src.corpus import byte_token_map                             # noqa: E402
from src.construct import identity, norm_match, cov_recolor, ridge_to_bpe  # noqa: E402
from src.learned import PatchProjection, byte_logprobs            # noqa: E402

tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
pe = PatchEmbedder(model, tok)
head = ByteHead(tok)

buf = []
for name in sorted(os.listdir("/content/qt/hnet-kaggle/docs")):
    if name.endswith(".md"):
        buf.append(open(os.path.join("/content/qt/hnet-kaggle/docs", name), "rb").read())
joined = b"\n".join(buf)
docs = [joined[i:i + 8192] for i in range(0, len(joined) - 8192, 8192)][:16]
train_docs, eval_docs = docs[:-3], docs[-3:]
print(f"[corpus] {len(train_docs)} train + {len(eval_docs)} eval", flush=True)

emb = pe.emb.detach().float().cuda()
PAD = emb.shape[0]
emb_ext = torch.cat([emb, torch.zeros(1, emb.shape[1], device="cuda")])


def run_staircase(chunker, P, label):
    def prep(data):
        """base sums (M,D), scored chunk positions ms, target bytes bs, and the BPE-token-mean
        target vector for the ridge pairing."""
        spans = chunker.spans(data)
        ids = torch.full((len(spans), P), PAD, dtype=torch.long)
        for m, (_, c) in enumerate(spans):
            for i, b in enumerate(c[:P]):
                if b in bmap:
                    ids[m, i] = bmap[b]
        base = emb_ext[ids.cuda()].sum(1)
        ms, bs, tgt_ids = [], [], []
        for m in range(len(spans) - 1):
            b = int(spans[m + 1][1][0])
            if b in bmap:
                ms.append(m); bs.append(b)
        for _, c in spans:
            tids = tok.encode(c.decode("latin-1"), add_special_tokens=False)
            tgt_ids.append(tids if tids else [0])
        tgt = torch.stack([emb[torch.tensor(t, device="cuda")].mean(0) for t in tgt_ids])
        return base, torch.tensor(ms, device="cuda"), torch.tensor(bs, device="cuda"), tgt

    train = [prep(d) for d in train_docs]
    evals = [prep(d) for d in eval_docs]
    X = torch.cat([t[0] for t in train])
    Y = torch.cat([t[3] for t in train])
    print(f"\n=== {label}: {X.shape[0]} fit chunks ===", flush=True)

    maps = {
        "identity": identity(),
        "norm_match": norm_match(emb),
        "cov_recolor": cov_recolor(X, emb),
        "ridge_bpe": ridge_to_bpe(X, Y),
    }

    def eval_map(apply):
        tot, n = 0.0, 0
        with torch.no_grad():
            for base, ms, bs, _ in evals:
                patch = apply(base).to(torch.bfloat16).unsqueeze(0)
                logits = model(inputs_embeds=patch).logits[0].float()
                lp = byte_logprobs(logits, head.first)
                nll = -lp[ms, bs]
                tot += float(nll.sum()); n += len(bs)
        return tot / max(n, 1)

    print(f"{'map':>14} {'boundary bpb':>13}", flush=True)
    for name, f in maps.items():
        print(f"{name:>14} {eval_map(f):>13.4f}", flush=True)

    proj = PatchProjection(emb.shape[1], rank=32).cuda()
    opt = torch.optim.AdamW(proj.parameters(), lr=3e-3)
    for ep in range(12):
        for base, ms, bs, _ in [train[i] for i in
                                np.random.RandomState(ep).permutation(len(train))]:
            patch = proj(base).to(torch.bfloat16).unsqueeze(0)
            logits = model(inputs_embeds=patch).logits[0].float()
            lp = byte_logprobs(logits, head.first)
            loss = -lp[ms, bs].mean()
            opt.zero_grad(); loss.backward(); opt.step()
    for p in proj.parameters():
        p.requires_grad_(False)
    print(f"{'sgd_rank32':>14} {eval_map(proj):>13.4f}", flush=True)

    dW = (proj.A @ proj.B).detach().float()
    s = torch.linalg.svdvals(dW)
    print(f"  [autopsy] ||dW||/||I|| {float(dW.norm() / emb.shape[1] ** 0.5):.4f} "
          f"top sv {np.round(s[:4].cpu().numpy(), 3)}", flush=True)
    for name, f in maps.items():
        if name == "identity":
            continue
        try:
            Wc = f(torch.eye(emb.shape[1], device="cuda"))
            c = float(torch.nn.functional.cosine_similarity(
                Wc.flatten() - torch.eye(emb.shape[1], device="cuda").flatten(),
                dW.flatten(), dim=0))
            print(f"    cos(dW, {name}-I) = {c:.3f}", flush=True)
        except Exception as e:
            print(f"    {name}: {e}", flush=True)


run_staircase(PeriodicChunker(6), 6, "periodic-6")
run_staircase(SimChunker(pe, target_mean=6.0), 16, "sim~6")
