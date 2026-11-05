"""The family sweep: is byte-competence an existing-weights property across the Qwen3.5 family?

    colab run --gpu A100 kernel/family_sweep.py

Per model (same eval docs, same protocol): stock BPE bpb, byte-level eff bpb, and the
fully-constructed effective bpb (ridge->BPE patch map fit on markdown train docs, closed form).
Zero training anywhere — just different pretrained weights.  -Base variants test whether
instruct tuning hardened token-level behavior.
"""
import os, subprocess, sys, gc

REPO = "/content/qt/hnet-qwen-adapter"
MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-2B",
          "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]


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

from src.adapter import ByteHead, ChunkLM, PatchEmbedder, PeriodicChunker  # noqa: E402
from src.corpus import byte_token_map, ValDocs                    # noqa: E402
from src.construct import ridge_to_bpe                            # noqa: E402
from src.learned import byte_logprobs                             # noqa: E402

buf = []
for name in sorted(os.listdir("/content/qt/hnet-kaggle/docs")):
    if name.endswith(".md"):
        buf.append(open(os.path.join("/content/qt/hnet-kaggle/docs", name), "rb").read())
joined = b"\n".join(buf)
docs = [joined[i:i + 8192] for i in range(0, len(joined) - 8192, 8192)][:16]
train_docs, eval_docs = docs[:-3], docs[-3:]
vdocs = ValDocs("/content/val").docs(8, min_len=4096) if os.path.isdir("/content/val") else []
P = 6
print(f"[corpus] {len(train_docs)} train + {len(eval_docs)} eval + {len(vdocs)} val", flush=True)


def eval_model(name):
    try:
        src = snapshot_download(name)
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
        for p in model.parameters():
            p.requires_grad_(False)
    except Exception as e:
        print(f"{name:>28}  LOAD FAIL: {type(e).__name__}", flush=True)
        return

    try:
        bmap = byte_token_map(tok)
        pe = PatchEmbedder(model, tok)
        head = ByteHead(tok)
        emb = pe.emb.detach().float().cuda()
        PAD = emb.shape[0]
        emb_ext = torch.cat([emb, torch.zeros(1, emb.shape[1], device="cuda")])
        chunker = PeriodicChunker(P)
        lm1 = ChunkLM(model, tok, period=1, agg="sum", byte_head=head)

        def base_sums(spans):
            ids = torch.full((len(spans), P), PAD, dtype=torch.long)
            for m, (_, c) in enumerate(spans):
                for i, b in enumerate(c[:P]):
                    if b in bmap:
                        ids[m, i] = bmap[b]
            return emb_ext[ids.cuda()].sum(1)

        # ridge fit on markdown train docs (closed form, model-specific embeddings)
        Xs, Ys = [], []
        for d in train_docs:
            sp = chunker.spans(d)
            Xs.append(base_sums(sp))
            for _, c in sp:
                tids = tok.encode(c.decode("latin-1"), add_special_tokens=False) or [0]
                Ys.append(emb[torch.tensor(tids, device="cuda")].mean(0))
        ridge = ridge_to_bpe(torch.cat(Xs), torch.stack(Ys))

        def metrics(ds):
            """(stock_bpb, byte_eff, ridge_eff) on a doc list — all forward-only."""
            stock_bits = byte_eff = ridge_bits = tot_bytes = 0.0
            for d in ds:
                tot_bytes += len(d)
                ids = torch.tensor(tok.encode(d.decode("latin-1"),
                                                add_special_tokens=False))
                with torch.no_grad():
                    lg = model(input_ids=ids[None].cuda()).logits[0].float()
                stock_bits += float(-lg.log_softmax(-1)[:-1, ids[1:].cuda()].sum())
                del lg
                fb = lm1.byte_nll(d)                           # byte-level per-position NLL
                byte_eff += 8.0 + float(fb.sum())
                sp = chunker.spans(d)
                base = base_sums(sp)
                ms, bs, pos = [], [], []
                for m in range(len(sp) - 1):
                    b = int(sp[m + 1][1][0])
                    if b in bmap:
                        ms.append(m); bs.append(b); pos.append(sp[m + 1][0])
                if ms:
                    with torch.no_grad():
                        patch = ridge(base).to(torch.bfloat16).unsqueeze(0)
                        lg = model(inputs_embeds=patch,
                                   logits_to_keep=torch.tensor(ms, device="cuda")
                                   ).logits[0].float()
                    lp = byte_logprobs(lg, head.first)
                    nll = -lp[torch.arange(len(bs), device="cuda"),
                              torch.tensor(bs, device="cuda")]
                    ridge_bits += 8.0 + float(fb.sum() - fb[np.array(pos)].sum()
                                              + float(nll.sum()))
                    del lg, lp
                torch.cuda.empty_cache()
            # fb/nll are already bits; only stock is nats
            return (stock_bits / np.log(2) / tot_bytes, byte_eff / tot_bytes,
                    ridge_bits / tot_bytes)

        s, b, r = metrics(eval_docs)
        sv, bv, rv = metrics(vdocs) if vdocs else (float("nan"),) * 3
        print(f"{name:>28}  md: stock {s:.4f} byte {b:.4f} ridge-eff {r:.4f} | "
              f"val: {sv:.4f} {bv:.4f} {rv:.4f}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"{name:>28}  OOM", flush=True)
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()


for m in MODELS:
    eval_model(m)
