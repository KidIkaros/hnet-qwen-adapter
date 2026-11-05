"""The instruct<->Base interpolation sweep: W(a) = (1-a)*W_instruct + a*W_Base.

    colab run --gpu A100 kernel/merge_sweep.py

Same shape, same lineage (instruct is fine-tuned from Base) — the linear-mode-connectivity
regime where weight interpolation is legal.  Per alpha: stock BPE + byte-level eff on
eval docs and decontam docs.  Does byte-competence recover linearly toward Base?
"""
import os, subprocess, sys, glob

REPO = "/content/qt/hnet-qwen-adapter"
A_NAME, B_NAME = "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B-Base"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


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
from src.adapter import ByteHead, ChunkLM                        # noqa: E402
from src.corpus import byte_token_map, ValDocs                   # noqa: E402

buf = []
for name in sorted(os.listdir("/content/qt/hnet-kaggle/docs")):
    if name.endswith(".md"):
        buf.append(open(os.path.join("/content/qt/hnet-kaggle/docs", name), "rb").read())
joined = b"\n".join(buf)
eval_docs = [joined[i:i + 8192] for i in range(0, len(joined) - 8192, 8192)][-3:]
ddocs = [open(f, "rb").read() for f in sorted(glob.glob("data/decontam/*.bin"))][:8]
vdocs = ValDocs("/content/val").docs(8, min_len=4096) if os.path.isdir("/content/val") else []

src_a, src_b = snapshot_download(A_NAME), snapshot_download(B_NAME)
tok = AutoTokenizer.from_pretrained(src_a)
ma = AutoModelForCausalLM.from_pretrained(src_a, dtype=torch.bfloat16).cuda().eval()
mb = AutoModelForCausalLM.from_pretrained(src_b, dtype=torch.bfloat16).cuda().eval()
sd_a = {k: v.detach().clone().float() for k, v in ma.state_dict().items()}
sd_b = {k: v.detach().clone().float().cpu() for k, v in mb.state_dict().items()}
del mb
head = ByteHead(tok)
print(f"[merge] {A_NAME} x {B_NAME}", flush=True)


def metrics(ds):
    stock_bits = byte_bits = tot_bytes = 0.0
    for d in ds:
        tot_bytes += len(d)
        ids = torch.tensor(tok.encode(d.decode("latin-1"), add_special_tokens=False))
        with torch.no_grad():
            lg = ma(input_ids=ids[None].cuda()).logits[0].float()
        stock_bits += float(-lg.log_softmax(-1)[:-1].gather(
            1, ids[1:].cuda().unsqueeze(1)).sum())
        del lg
        fb = ChunkLM(ma, tok, period=1, agg="sum", byte_head=head).byte_nll(d)
        byte_bits += 8.0 + float(fb.sum())
        torch.cuda.empty_cache()
    return stock_bits / np.log(2) / tot_bytes, byte_bits / tot_bytes


for a in ALPHAS:
    sd = {k: ((1 - a) * sd_a[k] + a * sd_b[k].cuda()).to(torch.bfloat16) for k in sd_a}
    ma.load_state_dict(sd)
    s, b = metrics(eval_docs)
    sd_, bd_ = metrics(ddocs)
    sv, bv = metrics(vdocs) if vdocs else (float("nan"), float("nan"))
    print(f"alpha={a:>4.2f}  md: stock {s:.4f} byte {b:.4f} | decontam: {sd_:.4f} {bd_:.4f} | "
          f"val: {sv:.4f} {bv:.4f}", flush=True)
