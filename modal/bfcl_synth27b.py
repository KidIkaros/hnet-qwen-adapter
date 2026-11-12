"""BFCL end-to-end synth on the packed int4 27B — Modal H200.

Loads the GPTQ-int4 artifact (same loader as hnet_probe27b), wraps it in
a batched mean-logprob scorer, and runs kernel/bfcl_synth.run_all over
BFCL v3 categories. Every pipeline selection (tool / arg name / value)
is a closed-set argmax — no free generation — so the int4 syntax
weakness should not bind here; this measures selection+composition.

Receipt -> /artifacts/bfcl_synth_int4.json
"""
import modal

app = modal.App("bfcl-synth27b")
art = modal.Volume.from_name("hnet-artifacts", create_if_missing=True)
hfcache = modal.Volume.from_name("hnet-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "safetensors", "accelerate",
                 "huggingface_hub", "datasets", "numpy", "pyarrow", "pandas",
                 "zstandard", "flash-linear-attention")
    .add_local_dir("src", "/root/repo/src")
    .add_local_dir("kernel", "/root/repo/kernel")
)

INT4_DIR = "/artifacts/int4-27b-toolace"
MODEL = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored"
CHUNK = 8          # candidates per forward
MAX_CTX = 3072     # token cap on context (schema JSON can be huge)


@app.function(image=image, gpu="H200", timeout=4 * 3600, memory=196608,
              volumes={"/artifacts": art, "/hf": hfcache},
              secrets=[modal.Secret.from_name("hf-token")])
def synth_int4(cats="simple,multiple,parallel,parallel_multiple",
               n_per_cat=40, offset=0, max_calls=4, max_args=6,
               value_topk=0, ckpt_every=10):
    import os, sys, json, time
    os.environ["HF_HOME"] = "/hf"
    os.environ["REPO"] = "/root/repo"
    # scope knobs -> bfcl_synth module-level env reads (pre-import)
    os.environ.update({
        "CATS": cats, "N_PER_CAT": str(n_per_cat),
        "OFFSET": str(offset), "MAX_CALLS": str(max_calls),
        "MAX_ARGS": str(max_args), "VALUE_TOPK": str(value_topk),
        "CKPT_EVERY": str(ckpt_every),
        "CKPT_PATH": "/artifacts/bfcl_synth_partial.json",
    })
    sys.path.insert(0, "/root/repo")
    sys.path.insert(0, "/root/repo/kernel")
    import torch
    from src.int4_runtime import load_int4_model

    t0 = time.time()
    model, tok = load_int4_model(INT4_DIR)
    print(f"[synth] int4 model loaded ({time.time() - t0:.0f}s)", flush=True)

    # ---- batched mean-logprob scorer ----
    @torch.no_grad()
    def score_fn(ctx_text, cands):
        ctx_ids = tok(ctx_text, add_special_tokens=False,
                      return_tensors="pt").input_ids[0].tolist()
        if len(ctx_ids) > MAX_CTX:
            ctx_ids = ctx_ids[-MAX_CTX:]
        out = {}
        todo = [(c, tok(c, add_special_tokens=False,
                        return_tensors="pt").input_ids[0].tolist())
                for c in cands]
        # empty tokenizations score -inf; everything else batches by CHUNK
        todo = [(c, i) for c, i in todo if i]
        for s in range(0, len(todo), CHUNK):
            chunk = todo[s:s + CHUNK]
            maxlen = max(len(i) for _, i in chunk)
            B, T = len(chunk), len(ctx_ids) + maxlen
            ids = torch.full((B, T), tok.pad_token_id or 0,
                             dtype=torch.long)
            att = torch.zeros((B, T), dtype=torch.long)
            for r, (_, ci) in enumerate(chunk):
                row = ctx_ids + ci
                ids[r, :len(row)] = torch.tensor(row)
                att[r, :len(row)] = 1
            ids = ids.to("cuda"); att = att.to("cuda")
            logits = model(ids, attention_mask=att).logits.float()
            L = len(ctx_ids)
            for r, (c, ci) in enumerate(chunk):
                # positions L-1 .. L+len(ci)-2 predict cand tokens
                sl = logits[r, L - 1:L - 1 + len(ci)]
                tgt = torch.tensor(ci, device="cuda")
                lp = torch.log_softmax(sl, -1)
                out[c] = float(lp.gather(
                    1, tgt.unsqueeze(1)).mean().item())
        for c in cands:
            out.setdefault(c, -1e9)
        return out

    import bfcl_synth
    rows = bfcl_synth.run_all(score_fn, on_ckpt=art.commit)
    agg = bfcl_synth.aggregate(rows)
    rec = {"arm": "int4", "scope": {"cats": cats, "n_per_cat": n_per_cat,
                                   "offset": offset},
           "agg": agg, "rows": rows}
    json.dump(rec, open("/artifacts/bfcl_synth_int4.json", "w"), indent=1)
    art.commit()
    print(f"[synth] {agg}", flush=True)
    return agg


@app.local_entrypoint()
def main():
    import os
    call = synth_int4.spawn(
        cats=os.environ.get(
            "CATS", "simple,multiple,parallel,parallel_multiple"),
        n_per_cat=int(os.environ.get("N_PER_CAT", "40")),
        offset=int(os.environ.get("OFFSET", "0")),
        max_calls=int(os.environ.get("MAX_CALLS", "4")),
        max_args=int(os.environ.get("MAX_ARGS", "6")),
        value_topk=int(os.environ.get("VALUE_TOPK", "0")),
        ckpt_every=int(os.environ.get("CKPT_EVERY", "10")))
    print(f"spawned call_id={call.object_id}", flush=True)
