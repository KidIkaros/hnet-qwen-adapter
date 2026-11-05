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
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
src = snapshot_download(MODEL_NAME)

from src.adapter import ByteHead, PatchEmbedder, PeriodicChunker  # noqa: E402
from src.corpus import byte_token_map, ValDocs                    # noqa: E402
from src.construct import ridge_to_bpe                            # noqa: E402
from src.decoder import ByteDecoder, chunk_batch, pair_batch, SOS  # noqa: E402
from src.generate import generate                                 # noqa: E402
from src.learned import PatchProjection, byte_logprobs            # noqa: E402

tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
pe = PatchEmbedder(model, tok)
head = ByteHead(tok)

# ---- corpus: Nemotron-CC-v2.1 slice if present (Dec-2025 corpus), else fineweb-edu
import glob as _g                                                   # noqa: E402
nem = sorted(_g.glob("data/nemotron/*.bin"))
if nem:
    print("[corpus] Nemotron-CC-v2.1 slice", flush=True)
    nj = b"".join(open(f, "rb").read() for f in nem)
    split = int(len(nj) * 0.95)
    tr_j, te_j = nj[:split], nj[split:]
else:
    sh("pip", "install", "-q", "datasets")
    from datasets import load_dataset  # noqa: E402
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    tr_j, te_j = bytearray(), bytearray()
    for i, row in enumerate(fw):
        if i < 4000:
            tr_j += row["text"].encode() + b"\n\n"
        elif i < 4800:
            te_j += row["text"].encode() + b"\n\n"
        else:
            break
        if len(tr_j) > 12_000_000 and len(te_j) > 600_000:
            break
    tr_j, te_j = bytes(tr_j[:12_000_000]), bytes(te_j[:600_000])
train_docs = [tr_j[i:i + 8192] for i in range(0, len(tr_j) - 8192, 8192)][:600]
eval_docs = [te_j[i:i + 8192] for i in range(0, len(te_j) - 8192, 8192)]
wt_te = eval_docs[3:]
eval_docs = eval_docs[:3]
train_docs += [open(f, "rb").read() for f in
               __import__("glob").glob("data/decontam/*.bin")]
print(f"[corpus] {len(train_docs)} train + {len(eval_docs)} eval"
      f" + {len(wt_te)} held-test", flush=True)

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


def trunk_ctx(patch, grad=False):
    """(M, D) trunk last-hidden per chunk position; ctx for chunk j = hidden[j-1] (j=0 -> ctx0).
    grad=True keeps the graph for the projection's backward pass."""
    cm = torch.enable_grad() if grad else torch.no_grad()
    with cm:
        out = model(inputs_embeds=patch.unsqueeze(0).to(torch.bfloat16),
                    output_hidden_states=True)
        hs = out.hidden_states[-1][0].float()                   # (M, D)
        lg = out.logits[0]                                      # (M, V)
    ctx = torch.cat([torch.zeros(1, D, device=hs.device), hs[:-1]])
    return ctx, hs, lg


def ctx_pairs(ctx, ctx0):
    """(M,2,D): row m's pair window conditions the prefix half on ctx[m-1] (ctx0 for m=0) and this
    chunk's half on ctx[m]."""
    prev = torch.cat([ctx0.unsqueeze(0).unsqueeze(0), ctx[:-1].unsqueeze(1)])
    return torch.cat([prev, ctx.unsqueeze(1)], dim=1)


# ---- ridge->BPE initialization: the closed-form floor before any SGD
Xs, Ys = [], []
for d in train_docs:
    sp = spans_of(d)
    Xs.append(base_sums(sp))
    for _, c in sp:
        tids = tok.encode(c.decode("latin-1"), add_special_tokens=False) or [0]
        Ys.append(emb[torch.tensor(tids, device="cuda")].mean(0))
X = torch.cat(Xs); Y = torch.stack(Ys)
if len(X) > 300_000:                                          # ridge fit: subsample (closed-form)
    idx = torch.randperm(len(X), device="cuda")[:300_000]
    X, Y = X[idx], Y[idx]
Wr = ridge_to_bpe(X, Y)(torch.eye(D, device="cuda"))           # the closed-form map
dW = (Wr - torch.eye(D, device="cuda")).T                       # proj applies I + dW.T
U, S, Vt = torch.linalg.svd(dW)
r = 32
sq = S[:r].sqrt()
proj.A.data = U[:, :r] @ torch.diag(sq)
proj.B.data = torch.diag(sq) @ Vt[:r, :]
print(f"[init] ridge->BPE SVD-32 init (top sv {np.round(S[:4].cpu().numpy(),2)})", flush=True)

# ---- a short SGD pass on top of the closed-form floor (the 'margin' the training buys)
if True:
    opt = torch.optim.AdamW(proj.parameters(), lr=3e-3)
    for ep in range(6):
        for d in train_docs:
            sp = spans_of(d)
            base = base_sums(sp)
            patch = proj(base)
            ctx, hs, lg = trunk_ctx(patch, grad=True)
            lp = byte_logprobs(lg[:-1], head.first)             # pos j-1 predicts chunk j's byte0
            tgt = torch.tensor([sp[j + 1][1][0] for j in range(len(sp) - 1)], device="cuda")
            loss = -lp[torch.arange(len(tgt)), tgt].mean()
            opt.zero_grad(); loss.backward(); opt.step()
    for p in proj.parameters():
        p.requires_grad_(False)
    print("[v2] patch projection trained inline (6 epochs)", flush=True)

# ---- decoder training: the scaling ablation runs both capacities in one session
def full_byte_bpb(dec, ds, label):
    tot_bits = tot_bytes = 0.0
    for d in ds:
        sp = spans_of(d)
        base = base_sums(sp)
        ctx, hs, lg = trunk_ctx(proj(base))
        ctx2 = ctx_pairs(ctx, dec.ctx0)
        bits = 8.0                                             # byte 0: uniform
        lp = byte_logprobs(lg[:-1], head.first)                # boundary bytes by the trunk head
        for j in range(len(sp) - 1):
            bits += float(-lp[j, sp[j + 1][1][0]])
        x, y, pad, lm = [t.cuda() for t in pair_batch(d, sp, max_len=P)]
        with torch.no_grad():
            logits = dec(ctx2, x, pad)
        lsm = logits.float().log_softmax(-1)
        bits += float(-lsm[lm, y[lm]].sum())
        tot_bits += bits; tot_bytes += len(d)
    bpb = tot_bits / np.log(2) / tot_bytes
    print(f"    [v2] FULL-BYTE bpb on {label}: {bpb:.4f} ({tot_bytes:.0f} bytes)", flush=True)
    return bpb


vdocs = ValDocs("/content/val").docs(8, min_len=4096) if os.path.isdir("/content/val") else []

# ---- leaderboard floors on wikitext-103 test: stock + byte-level
from src.adapter import qwen_byte_bpb, ChunkLM                     # noqa: E402
lm1 = ChunkLM(model, tok, period=1, agg="sum", byte_head=head)
_sb = _bb = _tb = 0.0
for d in wt_te[3:]:
    qb, _, nb = qwen_byte_bpb(model, tok, d)
    _sb += qb * nb
    fb = lm1.byte_nll(d)
    _bb += 8.0 * len(d) + float(fb.sum())
    _tb += len(d)
print(f"[floor] wt-test stock {_sb / _tb:.4f} byte {_bb / _tb:.4f} ({int(_tb)} B)",
      flush=True)

DECS = {}
for d_dec, n_layers, epn in ((256, 4, 8), (384, 6, 12)):
    name = f"d{d_dec}x{n_layers}"
    dec = ByteDecoder(D, max_len=P, d_dec=d_dec, n_layers=n_layers, n_heads=4).cuda()
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3)
    t0 = time.time(); step = 0
    for ep in range(epn):
        ep_tot = ep_n = 0.0
        for i in np.random.RandomState(ep).permutation(len(train_docs)):
            d = train_docs[i]
            sp = spans_of(d)
            base = base_sums(sp)
            ctx, hs, lg = trunk_ctx(proj(base))
            ctx2 = ctx_pairs(ctx, dec.ctx0)
            x, y, pad, lm = [t.cuda() for t in pair_batch(d, sp, max_len=P)]
            logits = dec(ctx2, x, pad)
            loss = F.cross_entropy(logits[lm].float(), y[lm])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_tot += float(loss) * int(lm.sum()); ep_n += int(lm.sum()); step += 1
    tr = ep_tot / ep_n / np.log(2)
    print(f"[{name}] interior train bpb {tr:.4f} | {sum(p.numel() for p in dec.parameters()) / 1e6:.2f}M "
          f"params | {time.time() - t0:.0f}s", flush=True)
    full_byte_bpb(dec, eval_docs, "eval docs (wikitext held)")
    full_byte_bpb(dec, wt_te[3:], "WIKITEXT-103 TEST (leaderboard)")
    if vdocs:
        full_byte_bpb(dec, vdocs, "hnet val split (cross-domain)")
    DECS[name] = dec
    torch.save(dec.state_dict(), f"/content/byte_decoder_{name}.pt")

dec = DECS["d384x6"]

print("[refs] byte-level 5.57 | constructed-sum eff 5.56 | v1 eff 5.27 | stock BPE 4.92 "
      "(eval docs)", flush=True)

# ---- generation: the sampler, not just the scorer
for seed_text, tag in ((b"# Report\n\n## Introduction\n\n", "markdown"),
                       (b"The history of the", "english")):
    g = generate(model, dec, proj, head, emb_ext, bmap, trunk_ctx, ctx_pairs,
                 seed=seed_text, n_chunks=8, P=P, pad_id=PAD)
    print(f"\n[generate:{tag}] {g.decode('latin-1')!r}", flush=True)
torch.save(dec.state_dict(), "/content/byte_decoder.pt")
print("[save] /content/byte_decoder.pt", flush=True)
