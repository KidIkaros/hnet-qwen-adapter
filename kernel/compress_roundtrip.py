"""End-to-end artifact test: quantize -> pack -> unpack -> verify.

GPTQ-quantizes the trunk (no rotation, validated config), entropy-packs
every weight matrix's codes with rANS, unpacks them, and checks exact
reconstruction of the dequantized weights. Reports the packed size vs
fp16 — the real 10% accounting, not just bits/weight estimates.

    colab exec -s hrun7 -f kernel/compress_roundtrip.py
    env: BITS (3), N_CALIB (24), ROTATE (0)
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

import numpy as np                                                  # noqa: E402
import torch                                                       # noqa: E402
from src.probe_common import (RESULTS, build_model, forward_bp,    # noqa: E402
                              load_ckpt, stream_toolace)
from src.compress import calibrate_hessians, quantize_linear       # noqa: E402
from src.pack import rans_encode, rans_decode                      # noqa: E402

BITS = int(os.environ.get("BITS", "3"))
N_CALIB = int(os.environ.get("N_CALIB", "24"))
ROTATE = os.environ.get("ROTATE", "0") == "1"

trunk, hn, meta = build_model()
load_ckpt(hn)
hn.eval()

targets = [(n, m) for n, m in hn.named_modules()
           if isinstance(m, torch.nn.Linear)
           and n.split(".")[-1] in ("q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj")]
calib = stream_toolace(1224, N_CALIB)
calib_ids = [list(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [".encode())
             for it in calib]
H = calibrate_hessians(hn, [n for n, _ in targets], calib_ids,
                       lambda ids: forward_bp(hn, trunk, ids, cap=256))

fp16_bytes = sum(m.weight.numel() * 2 for _, m in targets)
packed_bytes = 0
max_err = 0.0
qmax = 2 ** BITS - 1
t0 = time.time()
with torch.no_grad():
    for name, lin in targets:
        h = H[name]
        if h is None:
            continue
        W0 = lin.weight.data.clone()
        info = quantize_linear(lin, h.to(lin.weight.device),
                               bits=BITS, rotate=ROTATE)
        Wq = lin.weight.data.clone()
        # codes from the dequantized weights
        s = info["scale"].to(Wq.device).clamp(min=1e-8)
        z = info["zero"].to(Wq.device)
        codes = ((Wq / s) + z).round().clamp(0, qmax).to(torch.int64)
        data, tab = rans_encode(codes.cpu().numpy(), qmax + 1)
        dec = rans_decode(data, tab)
        codes_rt = torch.tensor(dec.reshape(codes.shape),
                                dtype=torch.int64, device=Wq.device)
        rec = (codes_rt.float() - z) * s
        max_err = max(max_err, float((rec - Wq).abs().max()))
        packed_bytes += len(data)
        lin.weight.data.copy_(W0)          # restore, no cross-layer drift

print(f"[artifact] {BITS}bit | fp16 {fp16_bytes/2**20:.0f}MiB | "
      f"packed {packed_bytes/2**20:.0f}MiB | ratio "
      f"{packed_bytes/fp16_bytes:.3f} | recon max_err {max_err:.2e} | "
      f"{time.time()-t0:.0f}s", flush=True)

res = {"bits": BITS, "fp16_mib": round(fp16_bytes / 2**20, 1),
       "packed_mib": round(packed_bytes / 2**20, 1),
       "ratio": round(packed_bytes / fp16_bytes, 4),
       "recon_max_err": max_err, "n_targets": len(targets)}
json.dump(res, open(f"{RESULTS}/artifact_{int(time.time())}.json", "w"))
print("[artifact] complete", flush=True)
