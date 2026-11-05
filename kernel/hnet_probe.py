"""Boundary-rule sweep on the H-Net-of-Qwen: periodic vs spacelike vs sim.

    colab run --gpu A100 kernel/hnet_probe.py

Real text (decontam doc), all boundary rules free/zero-training:
  periodic-6 | spacelike (SpaceByte: boundary after non-letter bytes) | sim
Scores byte bpb under full/sparse main x prev/retrieve ctx.
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

src = snapshot_download("Qwen/Qwen3.5-0.8B")
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
bmap = byte_token_map(tok)
hn = HNetQwen(model, tok, bmap, dec_layers=14).cuda()
hn.dechunk.weight.data.zero_()          # zero-init: ctx is a no-op at t=0
head = ByteHead(tok)
from src.learned import byte_logprobs                          # noqa: E402

data = open(sorted(glob.glob("data/decontam/github_*.bin"))[0], "rb").read()[:8192]
byte_ids = torch.tensor(list(data), device="cuda")
T = len(byte_ids)
print(f"[probe] {T} real bytes", flush=True)


def score(logits):
    bp = byte_logprobs(logits[0].float(), head.first)
    nll = -bp[:-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    return float(nll.sum()) / np.log(2) / (T - 1)


def periodic(T_, P=6):
    b = torch.zeros(T_, device="cuda"); b[::P] = 1.0
    return b


def spacelike(data_, T_):
    """SpaceByte: boundary after a byte that is not letter/digit/continuation."""
    b = torch.zeros(T_, device="cuda")
    prev_space = True
    for i, x in enumerate(data_):
        is_space = not (48 <= x <= 57 or 65 <= x <= 90 or 97 <= x <= 122
                        or 128 <= x <= 191)
        if is_space and not prev_space:
            b[i] = 1.0
        prev_space = is_space
    b[0] = 1.0
    return b


def sim_rule(enc):
    """Boundary where cos-sim of adjacent encoder states is lowest — target ~6B."""
    q = enc / enc.norm(dim=-1, keepdim=True)
    s = (q[1:] * q[:-1]).sum(-1)                                # (T-1,) adjacency sim
    tau = s.mean() - 0.5 * s.std()
    b = torch.zeros(len(enc), device=enc.device)
    b[0] = 1.0
    b[1:][s < tau] = 1.0
    return b


# encoder states once for sim rule
tok_ids = torch.tensor([bmap.get(int(x), 0) for x in data], device="cuda")
hs = hn.emb(tok_ids)[None]
he = hs
for l in hn.enc:
    he = l(he, position_embeddings=None, attention_mask=None, position_ids=None)
sim_b = sim_rule(he[0])

rules = {"periodic-6": periodic(T), "spacelike": spacelike(data, T),
         "sim": sim_b}
modes = [(None, "prev", "full|prev"), (None, "retrieve", "full|retr"),
         ((32, 8, 4), "retrieve", "sparse|retr")]
for rname, bnd in rules.items():
    for sparse, ctx, tag in modes:
        with torch.no_grad():
            logits, _, spans = hn(byte_ids, boundaries=bnd, sparse=sparse, ctx=ctx)
        ml = np.mean([b - a for a, b in spans])
        print(f"  {rname:<10} {tag:<12} byte bpb {score(logits):.4f} "
              f"({len(spans)} ch, {ml:.1f}B)", flush=True)
