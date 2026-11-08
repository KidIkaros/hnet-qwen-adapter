"""Step 0: does a GPTQ-compressed trunk keep our behavior signals?

Loads the leg-9 2B + adapters, calibrates Hessians over ToolACE prompts,
GPTQ-quantizes the trunk linears (bits via env), then runs the same
40-item emit slice before/after. Records the ledger entry.

    colab exec -s hrun7 -f kernel/compress_check.py
    env: BITS (4), N_CALIB (64), N_ITEMS (40), VERIFY_SKIP (2500)
"""
import os, sys, subprocess, json, time

REPO = "/content/qt/hnet-qwen-adapter"


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-2500:], flush=True)
        raise SystemExit(f"FAILED {' '.join(cmd)}")
    return r.stdout


if os.path.exists(REPO):
    sh("git", "-C", REPO, "pull", "-q")
else:
    sh("git", "clone", "-q",
       "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)

import torch                                                       # noqa: E402
from src.probe_common import (RESULTS, build_model, dup_class,     # noqa: E402
                              forward_bp, greedy_emit, load_ckpt,
                              stream_toolace)
from src.compress import (calibrate_hessians, quantize_linear,     # noqa: E402
                          trunk_linears, wanda_prune)

BITS = int(os.environ.get("BITS", "4"))
ROTATE = os.environ.get("ROTATE", "1") == "1"
PRUNE = float(os.environ.get("PRUNE", "0"))
N_CALIB = int(os.environ.get("N_CALIB", "64"))
N_ITEMS = int(os.environ.get("N_ITEMS", "40"))
VSKIP = int(os.environ.get("VERIFY_SKIP", "2500"))

trunk, hn, meta = build_model()
load_ckpt(hn)
hn.eval()

items = stream_toolace(VSKIP, N_ITEMS)
prompts = [f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode()
           for it in items]


@torch.no_grad()
def run_emit():
    loops = closed = 0
    for pid in prompts:
        t = greedy_emit(hn, trunk, list(pid), 300)
        closed += b"]" in t
        loops += dup_class(t)[0] is not None
    return loops, closed


BASE_RUN = os.environ.get("BASE_RUN", "0") == "1"
if BASE_RUN:
    loops0, closed0 = run_emit()
else:
    loops0, closed0 = 21, 29      # known reference on this slice (SKIP 2500)
print(f"[base] loops {loops0}/{N_ITEMS} closed {closed0}", flush=True)

# ---- calibrate + quantize trunk linears ----
calib = stream_toolace(1224, N_CALIB)
calib_ids = [list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode())
             for it in calib]

targets = trunk_linears(hn)
print(f"[quant] {len(targets)} linear targets", flush=True)

H = calibrate_hessians(hn, [n for n, _ in targets], calib_ids,
                       lambda ids: forward_bp(hn, trunk, ids, cap=256))
meta_q = {}
with torch.no_grad():
    for name, lin in targets:
        h = H[name]
        if h is None:
            continue
        if PRUNE > 0:
            wanda_prune(lin, h.to(lin.weight.device), PRUNE)
        meta_q[name] = quantize_linear(lin, h.to(lin.weight.device),
                                       bits=BITS, rotate=ROTATE)
print(f"[quant] quantized {len(meta_q)} linears at {BITS} bits", flush=True)

loops1, closed1 = run_emit()
print(f"[quant] loops {loops1}/{N_ITEMS} closed {closed1}", flush=True)

res = {"bits": BITS, "n_targets": len(meta_q),
       "base_loops": loops0, "base_closed": closed0,
       "q_loops": loops1, "q_closed": closed1, "n": N_ITEMS}
json.dump(res, open(f"{RESULTS}/compress_check_{int(time.time())}.json",
                    "w"))
from src.edit_ledger import record                                # noqa: E402
record({"method": "gptq-compression",
        "target": f"trunk linears x{len(meta_q)} @ {BITS}bit",
        "teach_gain": f"loops {loops0}->{loops1}",
        "neighbor_cost": f"closed {closed0}->{closed1}",
        "n_eval": N_ITEMS,
        "receipt": f"compress_check_{int(time.time())}.json"})
print("[compress_check] complete", flush=True)
