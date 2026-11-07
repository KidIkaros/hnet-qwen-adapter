"""Delta graft: apply (instruct - base) task vector to the trunk layers
inside the H-Net, then re-run the toolace generation probe. NEGATIVE RESULT
at alpha=1.0 (0/15 all metrics — broke the adapter-trunk interface); kept
for masked/alpha variants.

    colab exec -s hrunX -f kernel/hnet_delta_graft.py
    env: MODEL, DONOR (default Qwen/Qwen3.5-2B), CKPT_PREFIX,
         ALPHA (1.0), N_ITEMS (15), GEN_LEN (512), SKIP (2500)
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
import re, json, time                                           # noqa: E402
import gc                                                        # noqa: E402
from transformers import AutoModelForCausalLM                    # noqa: E402
from src.probe_common import (RESULTS, build_model, forward_bp,  # noqa: E402
                              load_ckpt, stream_toolace)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B-Base")
DONOR = os.environ.get("DONOR", "Qwen/Qwen3.5-2B")
ALPHA = float(os.environ.get("ALPHA", "1.0"))

trunk, hn, meta = build_model()
load_ckpt(hn)

print(f"[graft] loading donor {DONOR} ...", flush=True)
donor = AutoModelForCausalLM.from_pretrained(
    DONOR, dtype=torch.bfloat16, device_map="cpu")
donor_sd = donor.state_dict()
base_sd = trunk["model"].state_dict()
applied, skipped = 0, 0
with torch.no_grad():
    for name, w in donor_sd.items():
        if not name.startswith("model.layers."):
            continue                       # skip embed/norm/lm_head
        b = base_sd.get(name)
        if b is None or b.shape != w.shape:
            skipped += 1
            continue
        parts = name.split(".")
        mod = trunk["model"]
        for p_ in parts[:-1]:
            mod = getattr(mod, p_) if not p_.isdigit() else mod[int(p_)]
        t = getattr(mod, parts[-1])
        if t.shape == w.shape:
            t.add_(ALPHA * (w.to(t.dtype).cuda() - t))
            applied += 1
print(f"[graft] applied delta to {applied} tensors, skipped {skipped} "
      f"(alpha={ALPHA})", flush=True)
del donor, donor_sd, base_sd
gc.collect(); torch.cuda.empty_cache()

N_ITEMS = int(os.environ.get("N_ITEMS", "15"))
GEN_LEN = int(os.environ.get("GEN_LEN", "512"))
SKIP = int(os.environ.get("SKIP", "2500"))
CALL_RE = re.compile(r"([A-Za-z_][\w]*)\s*\(")


@torch.no_grad()
def generate(prompt_ids, max_new):
    ids = list(prompt_ids)
    for _ in range(max_new):
        bp = forward_bp(hn, trunk, ids)
        ids.append(int(bp[0, -1].argmax()))
        if ids[-6:] == list(b"\n\nUSER:") or ids[-2:] == list(b"\n\n"):
            break
    return bytes(ids[len(prompt_ids):]).decode("utf-8", "replace")


items = stream_toolace(SKIP, N_ITEMS)
print(f"[exam] {len(items)} items", flush=True)

right_n = call_n = hall_n = 0
rows = []
for k, it in enumerate(items):
    pid = list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT:".encode())
    gen = generate(pid, GEN_LEN)
    calls = CALL_RE.findall(gen)
    right = it["gold"] in calls
    hall = [c for c in calls if c not in it["tools"]]
    right_n += right; call_n += bool(calls); hall_n += bool(hall)
    rows.append({"gold": it["gold"], "calls": calls, "right": right,
                 "hall": hall, "gen_head": gen[:200]})
    print(f"  item {k:2d} gold={it['gold'][:20]} calls={calls[:3]} "
          f"{'RIGHT' if right else 'miss'} hall={hall[:2]}", flush=True)

n = len(items)
res = {"donor": DONOR, "alpha": ALPHA, "right": right_n, "any": call_n,
       "hall": hall_n, "n": n, "applied": applied, "rows": rows}
os.makedirs(RESULTS, exist_ok=True)
rp = f"{RESULTS}/graft_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
json.dump(res, open(rp, "w"), indent=1)
print(f"[graft] right {right_n}/{n} any {call_n}/{n} hall {hall_n}/{n} "
      f"(base probe was 4/12/8)", flush=True)
print(f"[receipt] -> {rp}", flush=True)
print("[graft] complete", flush=True)
