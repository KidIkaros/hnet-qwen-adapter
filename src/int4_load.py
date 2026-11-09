"""Load a packed-int4 GPTQ artifact (save_int4.py output) into a model.

The artifact dir contains:
  int4-XXXXX-of-YYYYY.safetensors  — nibble-packed codes (.qweight, uint8)
                                     + fp16 scales (.scales) per linear,
                                     bf16 for everything else
  INT4_MANIFEST.json               — bits/blocksize/sym/packing recipe
  config.json / tokenizer files    — copied from the bf16 artifact

Usage:
    from src.int4_load import load_int4
    model, tok = load_int4("/content/qt/int4-27b-toolace",
                           "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored")

Strategy: load the bf16 reference model, then overwrite every quantized
linear's weight with dequant_int(packed, scales). Dequantized weights are
bf16 — runtime cost identical to the bf16 artifact; the int4 format is a
STORAGE/TRANSFER optimization (load-time dequant), not a kernel optimization.
"""
import glob
import json
import os

import torch


def _shard_files(d):
    fs = sorted(glob.glob(os.path.join(d, "int4-*.safetensors")))
    if not fs:
        raise FileNotFoundError(f"no int4-*.safetensors in {d}")
    return fs


def load_packed_state(d, device="cpu"):
    """Read all shards -> {key: cpu tensor}."""
    from safetensors import safe_open
    out = {}
    for f in _shard_files(d):
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


@torch.no_grad()
def load_int4(model_dir, base_model=None, device_map="auto",
              max_memory=None, dtype=torch.bfloat16):
    """Build a transformers model whose linear weights come from int4 codes.

    model_dir: packed artifact dir. base_model: HF id or path of the
    unquantized reference (needed for architecture + non-quantized params
    not present... actually all non-target params ARE in the shards, but
    the model skeleton/config must come from somewhere — we load the
    reference config then inject our state dict).
    """
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoConfig)
    from src.compress import dequant_int

    man = json.load(open(os.path.join(model_dir, "INT4_MANIFEST.json")))
    bits = man.get("bits", 4)
    blocksize = man.get("packing", {}).get("blocksize", 128)
    base_model = base_model or man.get("base_model")

    sd = load_packed_state(model_dir)

    # split packed vs plain tensors
    qw = {k[:-8]: v for k, v in sd.items() if k.endswith(".qweight")}
    sc = {k[:-7]: v for k, v in sd.items() if k.endswith(".scales")}
    plain = {k: v for k, v in sd.items()
             if not (k.endswith(".qweight") or k.endswith(".scales"))}

    # dequantize every packed linear -> bf16 weight
    full_sd = dict(plain)
    for name, pk in qw.items():
        w = dequant_int(pk, sc[name].float(),
                        in_f=pk.shape[1] * 2, bits=bits,
                        blocksize=blocksize)
        # in_f recovered: packed width = in/2 -> in = pk.shape[1]*2
        full_sd[f"{name}.weight"] = w.to(dtype).contiguous()

    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_config(cfg)
    missing, unexpected = model.load_state_dict(full_sd, strict=False)
    real_missing = [k for k in missing
                    if not k.endswith((".bias",))]
    if real_missing:
        print(f"[int4_load] WARNING missing keys: {real_missing[:8]}")
    if unexpected:
        print(f"[int4_load] WARNING unexpected keys: {unexpected[:8]}")

    if device_map:
        model = model.to(dtype)
        from accelerate import dispatch_model          # noqa: E402
        dispatch_model(model, device_map=device_map)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    return model, tok
