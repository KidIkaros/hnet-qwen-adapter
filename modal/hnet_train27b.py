"""Modal app: H-Net byte-adapter TRAINING on the packed-int4 27B trunk.

Phase B — gradient fine-tune following the proven 2B recipe (hnet_train.py):
  enc/dec DeltaNet slices + dechunk + boundary + w_sup trainable;
  w_in/w_out/w_main/w_enc stay fixed ridge maps; trunk frozen.

    modal deploy modal/hnet_train27b.py
    modal run --detach modal/hnet_train27b.py   # env-tunable via entrypoint

Checkpoints -> /artifacts/hnet27b_{glue,enc,dec}.pt + hnet27b_best_*
Receipts -> /artifacts/hnet27b_receipt.json + RESULT_JSON.
"""
import modal

app = modal.App("hnet-train27b")
art = modal.Volume.from_name("hnet-artifacts", create_if_missing=True)
hfcache = modal.Volume.from_name("hnet-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "safetensors", "accelerate",
                 "huggingface_hub", "datasets", "numpy", "pyarrow", "pandas",
                 "zstandard", "flash-linear-attention")
    .add_local_dir("src", "/root/repo/src")
    .add_local_dir("data", "/root/repo/data")
)

MODEL = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored"
INT4_DIR = "/artifacts/int4-27b-toolace"


@app.function(image=image, gpu="H200", timeout=8 * 3600, memory=180000,
              volumes={"/artifacts": art, "/hf": hfcache},
              secrets=[modal.Secret.from_name("hf-token")])
def run_train(n_train=1000, epochs=1, lr=1e-4, enc_layers=4, dec_layers=4,
              eval_every=100, save_every=100, doc_cap=4096,
              ew_bytes=50000, aux_w=1.0):
    import os, sys, json, time, gc, glob
    os.environ["HF_HOME"] = "/hf"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    import torch
    import numpy as np
    from src.compress import dequant_int
    from src.int4_load import load_packed_state

    # ---------- load int4 trunk ------------------------------------------
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
    model.load_state_dict(full, strict=False)
    del full
    gc.collect()
    model = model.to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(INT4_DIR)
    print(f"[load] int4 trunk ready ({time.time() - t0:.0f}s)", flush=True)

    # ---------- H-Net build + adapter fit --------------------------------
    from src.corpus import byte_token_map
    from src.adapter import ByteHead
    from src.hnet import HNetQwen
    from src.factory import (build_hnet, fit_adapters, doc_bpb, enwik8,
                             periodic, corpus_stream, DOLMA_MIX)

    text = model.model if hasattr(model, "model") else model.language_model
    if not hasattr(text, "layers"):
        text = text.language_model
    bmap = byte_token_map(tok)
    trunk = {"model": model, "tok": tok, "bmap": bmap,
             "head": ByteHead(tok), "text": text}
    hn, meta = build_hnet(trunk, enc_layers=enc_layers, dec_layers=dec_layers)
    for layer in list(hn.enc) + list(hn.dec):
        layer.gradient_checkpointing = True
    print(f"[hnet] enc {enc_layers} dec {dec_layers} "
          f"main {len(hn.main)}", flush=True)

    all_docs = [open(f, "rb").read()
                for f in sorted(glob.glob("data/decontam/*.bin"))]
    fit_docs = [d[:doc_cap] for d in all_docs[:4]]
    held_docs = [d[:doc_cap] for d in all_docs[-8:]]
    ew = enwik8()
    ew_test = torch.tensor(list(ew["test"][:ew_bytes]), device="cuda")

    t0 = time.time()
    g = fit_adapters(hn, trunk, meta, fit_docs, P=6)
    print(f"[fit] {g} dechunk pairs ({time.time() - t0:.0f}s)", flush=True)

    train_docs = corpus_stream(DOLMA_MIX, n_train, cap=doc_cap)
    print(f"[data] {len(train_docs)} train docs", flush=True)

    # ---------- trainable set --------------------------------------------
    trainable = list(hn.enc.parameters()) + list(hn.dec.parameters()) + \
        list(hn.dechunk.parameters()) + list(hn.boundary.parameters()) + \
        list(hn.w_sup.parameters())
    for p in trainable:
        p.requires_grad_(True)
    n_train_p = sum(p.numel() for p in trainable)
    print(f"[train] {n_train_p / 1e6:.1f}M trainable | lr {lr}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    P = 6

    def doc_loss(byte_ids):
        T = len(byte_ids)
        bp, _, _ = hn(byte_ids, boundaries=periodic(T, P), ctx="retrieve",
                      n_ctx=1, super_k=4, head_first=trunk["head"].first)
        nll = -bp[0, :-1].gather(1, byte_ids[1:].long().unsqueeze(1)) \
            .squeeze(1)
        aux = (hn.last_p.mean() - 1.0 / P) ** 2
        return nll.mean() + aux_w * aux

    def eval_bpb(byte_ids):
        with torch.no_grad():
            return doc_bpb(hn, byte_ids, trunk["head"], periodic(
                len(byte_ids), P), ctx="retrieve", n_ctx=1, super_k=4)

    def save_ckpt(prefix):
        torch.save({"w_sup": hn.w_sup.state_dict(),
                    "dechunk": hn.dechunk.state_dict(),
                    "boundary": hn.boundary.state_dict()},
                   f"/artifacts/{prefix}_glue.pt")
        torch.save({f"enc_{i}": l.state_dict()
                    for i, l in enumerate(hn.enc)},
                   f"/artifacts/{prefix}_enc.pt")
        torch.save({f"dec_{i}": l.state_dict()
                    for i, l in enumerate(hn.dec)},
                   f"/artifacts/{prefix}_dec.pt")
        art.commit()
        print(f"  [ckpt] saved {prefix}", flush=True)

    receipts = {"n_train_pM": round(n_train_p / 1e6, 1),
                "n_train_docs": len(train_docs), "lr": lr,
                "enc": enc_layers, "dec": dec_layers,
                "n_fit_pairs": g}
    best_held = float("inf")
    step = 0
    eval_count = 0
    final_step = len(train_docs) * epochs
    t_train = time.time()
    for ep in range(epochs):
        for d in train_docs:
            byte_ids = torch.tensor(list(d), device="cuda")
            loss = doc_loss(byte_ids)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                print(f"  step {step} loss {float(loss):.3f} "
                      f"|gn| {float(gn):.2f} "
                      f"({(time.time() - t_train) / step:.1f}s/step)",
                      flush=True)
            if step % eval_every == 0 or step == final_step:
                eval_count += 1
                hb = float(np.mean([eval_bpb(
                    torch.tensor(list(hd), device="cuda"))
                    for hd in held_docs[:3]]))
                ewv = eval_bpb(ew_test)
                receipts[f"step{step}"] = {"held": round(hb, 4),
                                           "enwik8": round(ewv, 4)}
                json.dump(receipts,
                          open("/artifacts/hnet27b_receipt.json", "w"),
                          indent=1)
                art.commit()
                print(f"  [eval] step {step} held {hb:.4f} "
                      f"enwik8 {ewv:.4f}", flush=True)
                if hb < best_held:
                    best_held = hb
                    save_ckpt("hnet27b_best")
                if eval_count % 3 == 0 or step == final_step:
                    save_ckpt("hnet27b_latest")
            del byte_ids, loss
            torch.cuda.empty_cache()

    receipts["final_held"] = round(best_held, 4)
    receipts["steps"] = step
    receipts["train_seconds"] = round(time.time() - t_train, 1)
    json.dump(receipts, open("/artifacts/hnet27b_receipt.json", "w"),
              indent=1)
    art.commit()
    print("RESULT_JSON " + json.dumps(receipts), flush=True)
    print("[train27b] complete", flush=True)


@app.local_entrypoint()
def main():
    import os
    call = run_train.spawn(
        n_train=int(os.environ.get("N_TRAIN", "1000")),
        epochs=int(os.environ.get("EPOCHS", "1")),
        lr=float(os.environ.get("LR", "1e-4")),
        enc_layers=int(os.environ.get("ENC_LAYERS", "4")),
        dec_layers=int(os.environ.get("DEC_LAYERS", "4")),
        eval_every=int(os.environ.get("EVAL_EVERY", "100")),
        save_every=int(os.environ.get("SAVE_EVERY", "100")),
        doc_cap=int(os.environ.get("DOC_CAP", "4096")),
        ew_bytes=int(os.environ.get("EW_BYTES", "50000")))
    print(f"spawned call_id={call.object_id}", flush=True)
