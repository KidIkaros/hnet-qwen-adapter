"""Regression tests for the depth-recovery probe notebook (kernel/depth_probe.py).

Specifically guards against the v1 failure where passing `inputs_embeds`
positionally into `twil.generate(...)` caused transformers 5.x to treat the
bfloat16 embeddings tensor as `input_ids` (`RuntimeError: Expected tensor for
argument #1 'indices' to have Long/Int; got BFloat16`).

Tests execute the actual built notebook cell functions against a tiny
synthetic `GraniteForCausalLM` (2 layers, dim=32, no downloaded weights).
"""
import json
import os
import re
import subprocess
import sys
import time

import pytest
import torch
from transformers import GraniteConfig, GraniteForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def built_nb():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, os.path.join(repo, "kernel/depth_probe.py")],
                   check=True)
    return json.load(open("/tmp/twil_depth_probe.ipynb"))


@pytest.fixture(scope="module")
def tiny_granite():
    torch.manual_seed(0)
    cfg = GraniteConfig(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    return GraniteForCausalLM(cfg).eval().to(torch.bfloat16)


def _load_helpers_ns(built_nb, tiny_granite):
    cell_src = None
    for c in built_nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if "def s2_generate_embeds" in s:
                cell_src = s
                break
    assert cell_src is not None, "helpers cell not found in built notebook"
    # Execute only up to the in-session smoke gate (which uses cuda:0)
    defs_only = cell_src.split("# Leg-0 smoke gate:")[0]
    ns = {
        "torch": torch,
        "time": time,
        "re": re,
        "twil": tiny_granite,
        "EMB": tiny_granite.get_input_embeddings(),
        "PAD_ID": 0,
        "norm_name": lambda s: re.sub(r"[^a-z0-9]+", "", (s or "").lower()),
    }
    exec(compile(defs_only, "cell4_helpers", "exec"), ns)
    return ns


def test_s2_generate_embeds_runs_on_granite(built_nb, tiny_granite):
    ns = _load_helpers_ns(built_nb, tiny_granite)
    pids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    state_vec = torch.randn(32, dtype=torch.float32)
    emb, attn = ns["build_conditioned_embeds"](pids, state_vec)
    assert emb.shape == (1, ns["N_SOFT"] + 5, 32)
    assert emb.dtype == torch.bfloat16
    assert attn.shape == (1, ns["N_SOFT"] + 5)
    new_toks, dt = ns["s2_generate_embeds"](emb, attn, max_new=4)
    assert new_toks.ndim == 1 and new_toks.numel() == 4
    assert new_toks.dtype == torch.long
    assert dt >= 0.0


def test_s2_generate_ids_slices_prompt_and_rejects_floats(built_nb, tiny_granite):
    ns = _load_helpers_ns(built_nb, tiny_granite)
    pids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    new_toks, _ = ns["s2_generate_ids"](pids, max_new=3)
    assert new_toks.shape == (3,)
    # Passing float embeddings to s2_generate_ids must fail immediately
    with pytest.raises(AssertionError):
        ns["s2_generate_ids"](torch.randn(1, 5, 32, dtype=torch.bfloat16))


def test_bundle_and_metrics_cells_execute_end_to_end(built_nb, tmp_path):
    tmp = str(tmp_path)
    c2 = "".join(built_nb["cells"][2]["source"]).replace("/kaggle/working", tmp)
    exec(compile(c2, "cell2_bundle", "exec"), {})
    items = json.load(open(f"{tmp}/items.json"))
    s1_rows = json.load(open(f"{tmp}/s1_rows.json"))
    b_rows = json.load(open(f"{tmp}/b_rows_v1.json"))
    assert len(items) == 100 and len(s1_rows) == 100 and len(b_rows) == 61
    c_rows = [dict(r, ok=(not r["s1_ok"]) if i % 2 == 0 else r["ok"])
              for i, r in enumerate(b_rows)]
    c9 = "".join(built_nb["cells"][9]["source"]).replace("/kaggle/working", tmp)
    ns = {"items": items, "S1_ROWS": s1_rows, "B_ROWS": b_rows,
          "C_ROWS": c_rows, "b_skipped": 0, "c_skipped": 39,
          "json": json, "os": os}
    exec(compile(c9, "cell9_metrics", "exec"), ns)
    m = json.load(open(f"{tmp}/depth_metrics.json"))
    st = json.load(open(f"{tmp}/run_status.json"))
    assert m["b_n"] == 61 and m["b_emitted_n"] == 26
    assert m["c_n"] == 61 and m["paired_n"] == 61
    assert st["arms"] == {"s1": "done", "b": "done", "c": "done"}
    assert os.path.exists(f"{tmp}/artifacts.zip")

