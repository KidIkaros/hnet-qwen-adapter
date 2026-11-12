"""Native int4 inference: load the packed GPTQ artifact WITHOUT dequantizing.

Same artifact as src/int4_load.py / src/int4_runtime.py, but quantized
nn.Linear modules are replaced by Int4Linear modules that keep nibble-packed
weights resident on-device and run the Triton GEMM from src/int4_triton.py.

Memory: ~4.5 bits/param for the 400 trunk linears (~11GB) + bf16 for
embed/lm_head/norms/conv/deltanet (~8GB) ~= 19-20GB total — fits 24GB cards
(L4/A10G/3090) and 2xT4 (32GB), where the bf16 dequant path needs ~54GB.

Multi-GPU: pass devices=['cuda:0','cuda:1']; transformer layers split evenly
across devices with tiny forward-pre-hooks moving boundary tensors. Single
device: devices='cuda' (or ['cuda:0']).

    from src.int4_native import load_int4_native
    model, tok = load_int4_native("/path/to/int4-27b-toolace",
                                  devices=["cuda:0", "cuda:1"])
"""
import json
import os
import re

import torch
from torch import nn

from src.int4_triton import int4_gemm


class Int4Linear(nn.Module):
    """Drop-in nn.Linear replacement computing y = x @ W_int4^T via Triton.

    qweight [out, in//2] uint8 and scales [out, nb] float are registered as
        non-persistent buffers so model.to(device) moves them but they stay
        out of state_dict (the packed artifact is the source of truth).
    """

    def __init__(self, in_features, out_features, qweight, scales,
                 bias=None, bits=4, blocksize=128, out_dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.blocksize = blocksize
        self.out_dtype = out_dtype
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        if bias is not None:
            self.register_parameter(
                "bias", nn.Parameter(bias.detach().clone(),
                                     requires_grad=False))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = int4_gemm(x, self.qweight, self.scales,
                      bits=self.bits, blocksize=self.blocksize)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(self.out_dtype)


def _split_packed(sd):
    qw = {k[:-8]: v for k, v in sd.items() if k.endswith(".qweight")}
    sc = {k[:-7]: v for k, v in sd.items() if k.endswith(".scales")}
    plain = {k: v for k, v in sd.items()
             if not (k.endswith(".qweight") or k.endswith(".scales"))}
    return qw, sc, plain


def _norm_devices(devices):
    if isinstance(devices, str):
        return [devices]
    return list(devices)


def _module_devices(model, devices):
    """Map every named module to one device: layers split evenly, embed +
    rotary on the first device, final norm + lm_head on the last."""
    n = len(devices)
    n_layers = len(model.model.layers)
    dev_of = {}
    for name, _ in model.named_modules():
        m = re.search(r"\.layers\.(\d+)", name)
        if m:
            i = int(m.group(1))
            dev = devices[min(i * n // n_layers, n - 1)]
        elif name in ("lm_head", "model.norm"):
            dev = devices[-1]
        else:
            dev = devices[0]
        dev_of[name] = dev
    return dev_of


def _param_device(param_name, dev_of):
    """Device for a state-dict key by walking up to its module."""
    name = param_name
    while name:
        if name in dev_of:
            return dev_of[name]
        name = name.rpartition(".")[0]
    return next(iter(dev_of.values()))


def _section_root_names(model):
    """Modules whose inputs can arrive on a different device than their
    weights: backbone children + each transformer layer."""
    names = {"model", "lm_head"}
    for name, _ in model.named_modules():
        if name == "model.embed_tokens" or name == "model.norm" or \
                name == "model.rotary_emb":
            names.add(name)
        elif re.fullmatch(r"model\.layers\.\d+", name):
            names.add(name)
    return names


def _boundary_hooks(model, dev_of):
    """Pre-hooks moving args+kwargs tensors (recursively) to each section
    root's device. Only section roots need hooks: tensors can only change
    device at backbone/layer boundaries, never inside a layer."""
    roots = _section_root_names(model)

    def make(device):
        def mv(t):
            if torch.is_tensor(t):
                return t.to(device)
            if isinstance(t, tuple):
                return tuple(mv(x) for x in t)
            if isinstance(t, list):
                return [mv(x) for x in t]
            if isinstance(t, dict):
                return {k: mv(v) for k, v in t.items()}
            return t

        def hook(module, args, kwargs):
            return (tuple(mv(a) for a in args),
                    {k: mv(v) for k, v in kwargs.items()})
        return hook

    hooked = 0
    for name in roots:
        mod = model.get_submodule(name)
        dev = dev_of.get(name)
        if dev is None:
            continue
        mod.register_forward_pre_hook(make(dev), with_kwargs=True)
        hooked += 1
    return hooked


def swap_quantized_linears(model, qw, sc, dev_of, bits=4, blocksize=128):
    """Replace every nn.Linear named in qw with an Int4Linear.

    Returns (swapped_names, unmatched_names).
    """
    swapped, missing = [], []
    named = dict(model.named_modules())
    for name, packed in qw.items():
        mod = named.get(name)
        if mod is None or not isinstance(mod, nn.Linear):
            missing.append(name)
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        device = _param_device(name, dev_of)
        scales = sc[name].to(device=device)
        if scales.dtype != torch.float16:
            scales = scales.half()
        new = Int4Linear(
            mod.in_features, mod.out_features,
            packed.to(device=device), scales,
            bias=mod.bias.data.to(device) if mod.bias is not None else None,
            bits=bits, blocksize=blocksize,
            out_dtype=mod.weight.dtype,
        )
        setattr(parent, leaf, new)
        swapped.append(name)
    return swapped, missing


def load_int4_native(int4_dir, devices="cuda", attn_implementation=None):
    """Build the CausalLM with quantized linears kept packed on-device(s).

    devices: 'cuda' / 'cuda:0' / ['cuda:0', 'cuda:1', ...]. With multiple
    devices the 64 layers split evenly; embed+rotary on first, norm+lm_head
    on last, boundary tensors moved by forward-pre-hooks.

    Returns (model.eval(), tokenizer).
    """
    from accelerate import init_empty_weights
    from transformers import (AutoConfig, AutoModelForCausalLM,
                              AutoTokenizer)
    from src.int4_load import load_packed_state

    man = json.load(open(os.path.join(int4_dir, "INT4_MANIFEST.json")))
    bits = man.get("bits", 4)
    blocksize = man.get("packing", {}).get("blocksize", 128)
    devices = _norm_devices(devices)

    sd = load_packed_state(int4_dir)
    qw, sc, plain = _split_packed(sd)
    del sd

    cfg = AutoConfig.from_pretrained(int4_dir)
    with init_empty_weights():
        kwargs = {}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_config(cfg, **kwargs)

    # init_empty_weights leaves non-persistent buffers (rotary inv_freq) as
    # meta tensors; .to() on meta silently yields garbage. Re-materialize
    # them by re-instantiating the rotary module on CPU.
    backbone = getattr(model, "model", model)
    rotary = getattr(backbone, "rotary_emb", None)
    if rotary is not None and hasattr(rotary, "inv_freq"):
        with torch.device("cpu"):
            fresh = type(rotary)(cfg)
        rotary.inv_freq = fresh.inv_freq
        if hasattr(rotary, "original_inv_freq"):
            rotary.original_inv_freq = fresh.original_inv_freq

    dev_of = _module_devices(model, devices)
    swapped, missing = swap_quantized_linears(
        model, qw, sc, dev_of, bits=bits, blocksize=blocksize)
    print(f"[int4_native] swapped {len(swapped)} linears on "
          f"{len(devices)} device(s); unmatched packed keys: {len(missing)}")
    if missing:
        print(f"[int4_native] first unmatched: {missing[:5]}")
    del qw, sc

    n_hooks = _boundary_hooks(model, dev_of) if len(devices) > 1 else 0

    # Materialize remaining (bf16) params directly on their assigned device.
    plain_dev = {k: v.to(device=_param_device(k, dev_of), non_blocking=True)
                 for k, v in plain.items()}
    del plain
    missing_keys, unexpected = model.load_state_dict(
        plain_dev, strict=False, assign=True)
    real_missing = [k for k in missing_keys
                    if not k.endswith(".bias")]
    if real_missing:
        print(f"[int4_native] WARNING missing keys: {real_missing[:8]}")
    if unexpected:
        print(f"[int4_native] WARNING unexpected keys: {unexpected[:8]}")

    if n_hooks:
        print(f"[int4_native] attached {n_hooks} boundary hooks")
    model.eval()
    tok = AutoTokenizer.from_pretrained(int4_dir)
    return model, tok
