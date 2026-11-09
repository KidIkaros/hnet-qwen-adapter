"""Modal app: H-Net byte-adapter refit on the packed-int4 27B trunk — Phase A.

Validates the full path end-to-end and produces the first 27B-trunk byte-bpb:
  load int4 artifact (dequant-on-load) -> HNetQwen re-slice (enc=6 dec=8,
  matching the proven 2B recipe) -> smoke forward -> fit_adapters (closed-
  form ridge glue) -> byte-bpb on held decontam docs + enwik8 test slice.

No gradient training yet — this is the cheap gate: if adapters+forward work
and bpb lands sane vs the 2B baseline (enwik8-test 1.4317), Phase B
(slice fine-tune) is justified.

Per AGENTS.md: deploy + spawn detached, results to Modal Volume + RESULT_JSON.
    modal deploy modal/hnet_refit.py   # or modal run --detach
"""
import modal

app = modal.App("hnet-refit")
art = modal.Volume.from_name("hnet-artifacts", create_if_missing=True)
hfcache = modal.Volume.from_name("hnet-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "safetensors", "accelerate",
                 "huggingface_hub", "datasets", "numpy", "pyarrow", "pandas",
                 "flash-linear-attention")
    .add_local_dir("src", "/root/repo/src")
    .add_local_dir("data", "/root/repo/data")
)

MODEL = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored"
INT4_DIR = "/artifacts/int4-27b-toolace"
ENC = int(__import__("os").environ.get("ENC_LAYERS", "6"))
DEC = int(__import__("os").environ.get("DEC_LAYERS", "8"))
N_FIT = int(__import__("os").environ.get("N_FIT", "12"))
EW_BYTES = int(__import__("os").environ.get("EW_BYTES", "50000"))
P = int(__import__("os").environ.get("PERIOD", "6"))


@app.function(image=image, gpu="H200", timeout=4 * 3600, memory=180000,
              volumes={"/artifacts": art, "/hf": hfcache},
              secrets=[modal.Secret.from_name("hf-token")])
def run_refit():
    import os, sys, json, time, gc, glob
    os.environ["HF_HOME"] = "/hf"
    sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    import torch
    import numpy as np
    from src.compress import dequant_int
    from src.int4_load import load_packed_state

    # ---------- load int4 trunk (same recipe as hnet_eval) ---------------
    t0 = time.time()
    man = json.load(open(f"{INT4_DIR}/INT4_MANIFEST.json"))
    bits = man.get("bits", 4)
    blocksize = man.get("packing", {}).get("blocksize", 128)
    sd = load_packed_state(INT4_DIR)
    qw = {k[:-8]: v for k, v in sd.items() if k.endswith(".qweight")}
    sc = {k[:-7]: v for k, v in sd.items() if k.endswith(".scales")}
    full = {k: v for k, v in sd.items()
            if not (k.endswith(".qweight") or k.endswith(".scales"))}
    for name, pk in qw.items():
        w = dequant_int(pk.to("cuda"), sc[name].to("cuda").float(),
                        pk.shape[1] * 2, bits=bits, blocksize=blocksize)
        full[f"{name}.weight"] = w.to(torch.bfloat16).cpu().contiguous()
        del w
        torch.cuda.empty_cache()
    del sd, qw, sc
    from transformers import AutoConfig, AutoModelForCausalLM, \
        AutoTokenizer
    cfg = AutoConfig.from_pretrained(INT4_DIR)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(full, strict=False)
    print(f"[load] missing={[k for k in missing if not k.endswith('.bias')][:5]}"
          f" unexpected={unexpected[:5]}", flush=True)
    del full
    gc.collect()
    model = model.to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(INT4_DIR)
    print(f"[load] int4 trunk ready ({time.time() - t0:.0f}s)", flush=True)

    # ---------- H-Net re-slice -------------------------------------------
    from src.corpus import byte_token_map
    from src.adapter import ByteHead
    from src.hnet import HNetQwen
    from src.factory import build_hnet, fit_adapters, doc_bpb, enwik8, \
        periodic

    text = model.model if hasattr(model, "model") else model.language_model
    if not hasattr(text, "layers"):
        text = text.language_model
    bmap = byte_token_map(tok)
    trunk = {"model": model, "tok": tok, "bmap": bmap,
             "head": ByteHead(tok), "text": text}
    hn, meta = build_hnet(trunk, enc_layers=ENC, dec_layers=DEC)
    n_lin = sum(1 for l in hn.layers if l.block_type == "linear_attention")
    n_full = sum(1 for l in hn.layers if l.block_type == "full_attention")
    print(f"[hnet] {len(hn.layers)} layers | lin {n_lin} full {n_full} | "
          f"enc {ENC} dec {DEC} main {len(hn.main)} | "
          f"dec_start {meta['dec_start']}", flush=True)

    # ---------- smoke forward (T=256) ------------------------------------
    t0 = time.time()
    doc = open(sorted(glob.glob("data/decontam/*.bin"))[0], "rb").read()[:256]
    byte_ids = torch.tensor(list(doc), device="cuda")
    b = torch.zeros(len(byte_ids), device="cuda"); b[::P] = 1.0
    with torch.no_grad():
        bp, bounds, spans = hn(byte_ids, boundaries=b, ctx="retrieve",
                               n_ctx=1, super_k=4,
                               head_first=trunk["head"].first)
    nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)).squeeze(1)
    bpb256 = float(nll.mean()) / np.log(2)
    print(f"[smoke] T=256 forward OK | {len(spans)} chunks | "
          f"bpb {bpb256:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # ---------- adapter fit ----------------------------------------------
    all_docs = [open(f, "rb").read() for f in
                sorted(glob.glob("data/decontam/*.bin"))]
    fit_docs = [d[:4096] for d in all_docs[:N_FIT]]
    held_docs = [d[:4096] for d in all_docs[38:44]]
    t0 = time.time()
    g = fit_adapters(hn, trunk, meta, fit_docs, P=P)
    print(f"[fit] adapters fit | {g} dechunk pairs ({time.time() - t0:.0f}s)",
          flush=True)

    # ---------- eval ------------------------------------------------------
    receipts = {"smoke_bpb256": round(bpb256, 4), "n_fit_pairs": g}
    for i, hd in enumerate(held_docs):
        bids = torch.tensor(list(hd), device="cuda")
        bnd = periodic(len(bids), P)
        v = doc_bpb(hn, bids, trunk["head"], bnd, ctx="retrieve",
                    n_ctx=1, super_k=4)
        receipts[f"held_{i}"] = round(v, 4)
        print(f"[eval] held[{i}] bpb {v:.4f}", flush=True)
        del bids
        torch.cuda.empty_cache()
    receipts["held_mean"] = round(float(np.mean(
        [receipts[f"held_{i}"] for i in range(len(held_docs))])), 4)

    ew = enwik8()
    ew_ids = torch.tensor(list(ew["test"][:EW_BYTES]), device="cuda")
    bnd = periodic(len(ew_ids), P)
    t0 = time.time()
    ew_bpb = doc_bpb(hn, ew_ids, trunk["head"], bnd, ctx="retrieve",
                     n_ctx=1, super_k=4)
    receipts["enwik8_test_bpb"] = round(ew_bpb, 4)
    receipts["enwik8_bytes"] = len(ew_ids)
    print(f"[eval] enwik8-test bpb {ew_bpb:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    receipts.update({"enc": ENC, "dec": DEC, "period": P,
                     "n_fit_docs": N_FIT, "artifact": "qwen27b-int4-toolace",
                     "baseline_2b_enwik8": 1.4317})
    json.dump(receipts, open("/artifacts/hnet_refit_receipt.json", "w"),
              indent=1)
    art.commit()
    print("RESULT_JSON " + json.dumps(receipts), flush=True)
    print("[refit] complete", flush=True)


@app.local_entrypoint()
def main():
    call = run_refit.spawn()
    print(f"spawned call_id={call.object_id}", flush=True)
