"""Probe the re-sliced H-Net-of-Qwen: end-to-end forward on 0.8B.

    colab run --gpu A100 kernel/hnet_probe.py

Checks: layer slicing counts, forward pass shape, span stats under
periodic boundaries, and rough byte bpb vs the byte-level floor.
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
from huggingface_hub import snapshot_download                    # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer     # noqa: E402
from src.corpus import byte_token_map                             # noqa: E402
from src.adapter import ByteHead                                  # noqa: E402
from src.hnet import HNetQwen                                     # noqa: E402

src = snapshot_download("Qwen/Qwen3.5-0.8B")
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)

bmap = byte_token_map(tok)
hn = HNetQwen(model, tok, bmap).cuda()
lin = sum(1 for l in hn.layers if l.block_type == "linear_attention")
full = sum(1 for l in hn.layers if l.block_type == "full_attention")
print(f"[slice] {len(hn.layers)} layers -> enc {len(hn.enc)} + main {len(hn.main)} "
      f"+ dec {len(hn.dec)}  (DeltaNet {lin} / attn {full})", flush=True)

data = (b"The quick brown fox jumps over the lazy dog. "
        b"H-Net routes bytes through a frozen trunk.\n") * 40
byte_ids = torch.tensor(list(data), device="cuda")
T = len(byte_ids)
print(f"[probe] doc {T} bytes", flush=True)

# periodic boundaries every 6 bytes as the control route
b6 = torch.zeros(T, device="cuda"); b6[::6] = 1.0
with torch.no_grad():
    logits, bnd, spans = hn(byte_ids, boundaries=b6)
print(f"[forward] logits {tuple(logits.shape)} | {len(spans)} chunks "
      f"(mean {np.mean([b - a for a, b in spans]):.1f} B)", flush=True)

# byte-bpb via marginalized head (same path as the other kernels)
head = ByteHead(tok)
from src.learned import byte_logprobs                          # noqa: E402
bp = byte_logprobs(logits[0].float(), head.first)               # (T, 256)
tgt = byte_ids[1:].long()
nll = -bp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)
print(f"[debug] bp {tuple(bp.shape)} nll mean {float(nll.mean()):.3f} nats "
      f"max {float(nll.max()):.1f}", flush=True)
print(f"[score] byte bpb {float(nll.sum()) / np.log(2) / (T - 1):.4f}", flush=True)

# random-init learned boundaries: span stats
with torch.no_grad():
    logits2, bnd2, spans2 = hn(byte_ids, boundaries=None)
print(f"[random-route] {len(spans2)} chunks "
      f"(mean span {np.mean([b - a for a, b in spans2]):.1f} B)", flush=True)
