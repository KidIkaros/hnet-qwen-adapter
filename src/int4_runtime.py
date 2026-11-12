"""Shared int4-trunk loader for Modal drivers.

Single home for the packed-GPTQ -> bf16 dequant -> transformers load
sequence, previously duplicated across modal/hnet_probe27b.py and
modal/bfcl_synth27b.py. Lives in src/ because every Modal image already
mounts it via add_local_dir("src", "/root/repo/src").

    from src.int4_runtime import load_int4_model
    model, tok = load_int4_model(int4_dir="/artifacts/int4-27b-toolace")
"""


def load_int4_model(int4_dir, device="cuda"):
    """Dequantize packed int4 shards and load into a bf16 CausalLM.

    Returns (model.eval()'d on `device`, tokenizer). Caller owns freezing
    parameters / downstream use.
    """
    import gc
    import json
    import torch
    from src.compress import dequant_int
    from src.int4_load import load_packed_state

    man = json.load(open(f"{int4_dir}/INT4_MANIFEST.json"))
    bits = man.get("bits", 4)
    blocksize = man.get("packing", {}).get("blocksize", 128)
    sd = load_packed_state(int4_dir)
    qw = {k[:-8]: v for k, v in sd.items() if k.endswith(".qweight")}
    sc = {k[:-7]: v for k, v in sd.items() if k.endswith(".scales")}
    full = {k: v for k, v in sd.items()
            if not (k.endswith(".qweight") or k.endswith(".scales"))}
    for name, pk in qw.items():
        w = dequant_int(pk.to(device), sc[name].to(device).float(),
                        pk.shape[1] * 2, bits=bits, blocksize=blocksize)
        full[f"{name}.weight"] = w.to(torch.bfloat16).cpu().contiguous()
        del w
        torch.cuda.empty_cache()
    del sd, qw, sc

    from transformers import (AutoConfig, AutoModelForCausalLM,
                              AutoTokenizer)
    cfg = AutoConfig.from_pretrained(int4_dir)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    model.load_state_dict(full, strict=False)
    del full
    gc.collect()
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(int4_dir)
    return model, tok
