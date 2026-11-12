"""Push H-Net leg-1 checkpoints + receipts from the Modal volume to HF.

The trained adapter weights (~12.4GB) currently live only on the
hnet-artifacts volume — losing it loses leg-1. This mirrors them to
KidIkaros/hnet27b-leg1 so leg-2 resume and future work don't depend on
volume persistence. Runs server-side: volume -> HF directly, no local
roundtrip.

    modal run --detach modal/hf_checkpoint_push.py
"""
import modal

app = modal.App("hf-checkpoint-push")
art = modal.Volume.from_name("hnet-artifacts", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "hf_transfer")
)

REPO = "KidIkaros/hnet27b-leg1"
FILES = [
    "hnet27b_latest_glue.pt", "hnet27b_latest_enc.pt",
    "hnet27b_latest_dec.pt",
    "hnet27b_best_glue.pt", "hnet27b_best_enc.pt",
    "hnet27b_best_dec.pt",
    "hnet27b_receipt.json",
]


@app.function(image=image, cpu=4, memory=16384, timeout=2 * 3600,
              volumes={"/artifacts": art},
              secrets=[modal.Secret.from_name("hf-token")])
def push():
    import os, glob
    from huggingface_hub import HfApi
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    done = []
    for f in FILES:
        p = f"/artifacts/{f}"
        if not os.path.exists(p):
            print(f"[push] MISSING {f}", flush=True)
            continue
        api.upload_file(path_or_fileobj=p, path_in_repo=f,
                        repo_id=REPO, repo_type="model")
        done.append(f)
        print(f"[push] {f} ok", flush=True)
    # small receipts worth keeping alongside
    for extra in glob.glob("/artifacts/*receipt*.json") + \
            glob.glob("/artifacts/bfcl_synth*.json"):
        base = os.path.basename(extra)
        if base not in FILES:
            try:
                api.upload_file(path_or_fileobj=extra, path_in_repo=base,
                                repo_id=REPO, repo_type="model")
                done.append(base)
            except Exception as e:
                print(f"[push] {base} failed: {e}", flush=True)
    print(f"[push] complete: {len(done)} files -> {REPO}", flush=True)
    return done


@app.local_entrypoint()
def main():
    print(push.remote())
