"""Weight autopsy: reverse-engineer two deltas — what SFT wrote into the
instruct model (donor - base) and what 9 legs of CPT wrote into our adapter
(trained enc/dec - base lin layers). Per-tensor Frobenius norm + SVD
concentration tells us WHERE the competence lives and whether it's a small
transplantable subspace.

    colab exec -s hrunX -f kernel/hnet_weight_autopsy.py
    env: MODEL, DONOR (default Qwen/Qwen3.5-2B), CKPT_PREFIX, TOPN (15)
"""
import os, sys, glob, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch                                                     # noqa: E402
import json, time                                                # noqa: E402
from transformers import AutoModelForCausalLM                    # noqa: E402
from src.probe_common import RESULTS, sync_repo                  # noqa: E402
from src.factory import build_hnet, load_trunk                   # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base")
DONOR = os.environ.get("DONOR", "Qwen/Qwen3.5-2B")
TOPN = int(os.environ.get("TOPN", "15"))

trunk = load_trunk(MODEL)
hn, meta = build_hnet(trunk, enc_layers=6, dec_layers=8)

PREFIX = os.environ.get("CKPT_PREFIX", "hnet_resume_best")
glue = enc = dec = None
for d in [f"{REPO}/results", "/content/ckpt"]:
    try:
        enc = torch.load(f"{d}/{PREFIX}_enc.pt", map_location="cuda")
        dec = torch.load(f"{d}/{PREFIX}_dec.pt", map_location="cuda")
        break
    except FileNotFoundError:
        continue
print(f"[ckpt] loaded {PREFIX}_* from {d}", flush=True)

base_sd = trunk["model"].state_dict()
donor_sd = AutoModelForCausalLM.from_pretrained(
    DONOR, dtype=torch.float32).state_dict()


def eff_rank(sv, frac=0.95):
    c = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    return int((c < frac).sum()) + 1


def autopsy(label, pairs):
    """pairs: (tensor_name, delta_tensor). Ranked profile + group totals."""
    rows = []
    for name, D in pairs:
        D2 = D.float()
        flat = D2.reshape(D2.shape[0], -1) if D2.dim() > 1 else D2.unsqueeze(0)
        sv = torch.linalg.svdvals(flat)
        rows.append({"name": name, "norm": float(D2.norm()),
                     "top_sv": sv[:5].tolist(), "erank95": eff_rank(sv),
                     "rank": sv.shape[0]})
    rows.sort(key=lambda r: -r["norm"])
    print(f"\n[{label}] top {TOPN} tensors by |delta|:", flush=True)
    for r in rows[:TOPN]:
        print(f"  {r['name'][:52]:52s} |d| {r['norm']:8.3f} "
              f"erank95 {r['erank95']:4d}/{r['rank']}", flush=True)
    groups = {}
    for r in rows:
        g = ("mlp" if "mlp" in r["name"] else
             "attn" if any(k in r["name"] for k in ("qkv", "q_proj", "k_proj",
             "v_proj", "o_proj", "attn")) else
             "conv" if "conv" in r["name"] else "other")
        groups[g] = groups.get(g, 0) + r["norm"]
    print(f"  [groups] {groups}", flush=True)
    return rows


# ---- delta 1: instruct SFT delta (model.layers.*)
sft = [("sft:" + n, (donor_sd[n].cpu() - base_sd[n].cpu()))
       for n in donor_sd if n.startswith("model.layers.")
       and n in base_sd and donor_sd[n].shape == base_sd[n].shape]
sft_rows = autopsy("SFT instruct-base", sft)

# ---- delta 2: our CPT adapter delta (trained enc/dec vs base lin layers)
lin = meta["lin_idx"]
cpt = []
for i, li in enumerate(lin[:6]):
    for k, v in enc[f"enc_{i}"].items():
        b = base_sd.get(f"model.layers.{li}.{k}")
        if b is not None and b.shape == v.shape:
            cpt.append((f"cpt:enc_{i}.{k}", v.cpu() - b.cpu()))
for i, li in enumerate(lin[-8:]):
    for k, v in dec[f"dec_{i}"].items():
        b = base_sd.get(f"model.layers.{li}.{k}")
        if b is not None and b.shape == v.shape:
            cpt.append((f"cpt:dec_{i}.{k}", v.cpu() - b.cpu()))
cpt_rows = autopsy("CPT adapter-base", cpt)

json.dump({"sft": sft_rows, "cpt": cpt_rows,
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(f"{RESULTS}/weight_autopsy.json", "w"))
print(f"\n[wsp] -> {RESULTS}/weight_autopsy.json", flush=True)
print("[autopsy] complete", flush=True)
