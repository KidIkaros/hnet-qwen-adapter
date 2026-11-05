"""v1 training run: the low-rank patch projection on the frozen Qwen stack (the ZHA recipe).

    colab run --gpu A100 kernel/train_projection.py

Setup: SimChunker boundaries are PRECOMPUTED per doc and fixed (only the patch map learns).  Loss =
the eval metric itself: byte-marginalized NLL of the next chunk's first byte.  Train/eval split over
the markdown docs; reports before (identity = sim~6 baseline) vs after on held-out docs.
"""
import os, subprocess, sys, time

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


print("[setup] cloning the repos", flush=True)
sh("git", "clone", "-q", "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sh("git", "clone", "-q", "https://github.com/KidIkaros/hnet-kaggle.git", "/content/qt/hnet-kaggle")
sys.path.insert(0, REPO)
os.chdir(REPO)

print("[setup] downloading Qwen/Qwen3.5-0.8B", flush=True)
t0 = time.time()
import torch                                                     # noqa: E402
import numpy as np                                               # noqa: E402
from huggingface_hub import snapshot_download                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
src = snapshot_download("Qwen/Qwen3.5-0.8B")
print(f"[setup] done in {time.time() - t0:.0f}s", flush=True)

from src.adapter import ByteHead, PatchEmbedder, SimChunker       # noqa: E402
from src.corpus import byte_token_map                             # noqa: E402
from src.learned import PatchProjection, byte_logprobs, doc_tensors  # noqa: E402

tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in model.parameters():                                     # the trunk stays frozen
    p.requires_grad_(False)

bmap = byte_token_map(tok)
pe = PatchEmbedder(model, tok)
head = ByteHead(tok)
print(f"[setup] byte-token map: {len(bmap)}/256", flush=True)

# ---- corpus: markdown docs, N train + 3 held-out eval
buf = []
for name in sorted(os.listdir("/content/qt/hnet-kaggle/docs")):
    if name.endswith(".md"):
        buf.append(open(os.path.join("/content/qt/hnet-kaggle/docs", name), "rb").read())
joined = b"\n".join(buf)
NDOCS = int(os.environ.get("NDOCS", "16"))
docs = [joined[i:i + 8192] for i in range(0, len(joined) - 8192, 8192)][:NDOCS]
n_eval = 3
train_docs, eval_docs = docs[:-n_eval], docs[-n_eval:]
print(f"[corpus] {len(train_docs)} train + {n_eval} eval docs", flush=True)

# ---- fixed boundaries + per-doc constants
TM = float(os.environ.get("TARGET_MEAN", "6"))
chunker = SimChunker(pe, target_mean=TM, max_len=16)
emb = pe.emb.detach().float().cuda()
PAD = emb.shape[0]
emb_ext = torch.cat([emb, torch.zeros(1, emb.shape[1], device="cuda")])


def prep(data):
    spans = chunker.spans(data)
    ids, ms, bs = doc_tensors(data, spans, bmap, PAD)
    base = emb_ext[ids.cuda()].sum(1)                            # (M, D) fp32, constant
    return base, ms.cuda(), bs.cuda()


train = [prep(d) for d in train_docs]
evals = [prep(d) for d in eval_docs]
ml = np.mean([len(d) / (b.shape[0]) for d, (b, _, _) in zip(docs, train + evals)])
print(f"[chunker] sim~{TM:g}: realized mean_len {ml:.2f}", flush=True)

D = emb.shape[1]
RANK = int(os.environ.get("RANK", "32"))
LR = float(os.environ.get("LR", "3e-3"))
EPOCHS = int(os.environ.get("EPOCHS", "12"))
proj = PatchProjection(D, rank=RANK).cuda()
opt = torch.optim.AdamW(proj.parameters(), lr=LR)


def evaluate(split):
    tot, n = 0.0, 0
    with torch.no_grad():
        for base, ms, bs in split:
            patch = proj(base).to(torch.bfloat16).unsqueeze(0)
            logits = model(inputs_embeds=patch).logits[0].float()
            lp = byte_logprobs(logits, head.first)
            nll = -lp[ms, bs]
            tot += float(nll.sum()); n += len(bs)
    return tot / max(n, 1), n


from src.adapter import ChunkLM, qwen_byte_bpb                    # noqa: E402

# the fair controls: stock Qwen and the byte-level adapter scored on the SAME eval docs
qb_tot = qb_n = 0.0
for d in eval_docs:
    qb, _, nb = qwen_byte_bpb(model, tok, d)
    qb_tot += qb * nb; qb_n += nb
lm1 = ChunkLM(model, tok, period=1, agg="sum", byte_head=head)
b1_tot = b1_n = 0.0
for d in eval_docs:
    b1, k1, _ = lm1.first_byte_bpb(d)
    b1_tot += b1 * k1; b1_n += k1
print(f"[control] stock BPE on eval docs: {qb_tot / qb_n:.4f} | byte-level period-1: "
      f"{b1_tot / b1_n:.4f}", flush=True)

eb0, en0 = evaluate(evals)
print(f"[v1] BEFORE: eval bpb {eb0:.4f} over {en0} targets (identity map = constructed-sum baseline)", flush=True)

t0 = time.time()
step = 0
for ep in range(EPOCHS):
    order = np.random.RandomState(ep).permutation(len(train))
    ep_tot = ep_n = 0.0
    for i in order:
        base, ms, bs = train[i]
        patch = proj(base).to(torch.bfloat16).unsqueeze(0)
        logits = model(inputs_embeds=patch).logits[0].float()
        lp = byte_logprobs(logits, head.first)
        loss = -lp[ms, bs].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        ep_tot += float(loss) * len(bs); ep_n += len(bs); step += 1
    print(f"[epoch {ep:02d}] train bpb {ep_tot / ep_n:.4f} ({step} steps)", flush=True)
print(f"[timing] training {time.time() - t0:.0f}s ({step} steps)", flush=True)

eb, en = evaluate(evals)
tb, tn = evaluate(train)
print(f"\n[v1] AFTER:  eval bpb {eb:.4f} over {en} targets | train bpb {tb:.4f} over {tn}", flush=True)
print("[refs] constructed sim~6+sum 6.3867 (8 shared docs) | byte-level 5.2250 | stock BPE 4.2560 | "
      "H-Net 1.4177", flush=True)
