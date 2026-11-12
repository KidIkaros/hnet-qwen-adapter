"""Regression tests for the depth-recovery probe notebook (kernel/depth_probe.py).

v1 failure guarded: passing `inputs_embeds` positionally into
`twil.generate(...)` made transformers 5.x treat the bfloat16 tensor as
`input_ids`. v3 drops the embeds path entirely (v2 showed identity
injection is destructive) — Arm T conditions via `text_conditioned_prompt`.

Tests execute the actual built notebook cell functions against a tiny
synthetic `GraniteForCausalLM` (2 layers, dim=32, no downloaded weights)
and a FakeTok for the prompt builder.
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
            if "def s2_generate_ids" in s:
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


class FakeTok:
    def apply_chat_template(self, msgs, **kw):
        self.kw = kw
        return msgs


def test_text_conditioned_prompt_injects_decision(built_nb, tiny_granite):
    ns = _load_helpers_ns(built_nb, tiny_granite)
    ft = FakeTok()
    ns["tok"] = ft
    it = {"user": "book me a flight to oslo",
          "tools": [{"name": "book_flight", "description": "d",
                     "parameters": {}}]}
    msgs = ns["text_conditioned_prompt"](it, "book_flight")
    assert ft.kw["enable_thinking"] is True
    assert ft.kw["add_generation_prompt"] is True
    content = msgs[0]["content"]
    assert content.startswith("book me a flight to oslo")
    assert 'chose "book_flight"' in content
    assert "Verify" in content
    msgs2 = ns["text_conditioned_prompt"](it, "cancel_flight")
    assert 'chose "cancel_flight"' in msgs2[0]["content"]


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
    cells = built_nb["cells"]
    bundle_src = metrics_src = None
    for c in cells:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "b_rows_v1.json" in s and "b64decode" in s:
            bundle_src = s
        if "t_vs_b_on_s1_wrong" in s:
            metrics_src = s
    assert bundle_src and metrics_src, "bundle or metrics cell not found"
    exec(compile(bundle_src.replace("/kaggle/working", tmp),
                 "bundle_cell", "exec"), {})
    items = json.load(open(f"{tmp}/items.json"))
    s1_rows = json.load(open(f"{tmp}/s1_rows.json"))
    b_rows = json.load(open(f"{tmp}/b_rows_v1.json"))
    assert len(items) == 100 and len(s1_rows) == 100 and len(b_rows) == 61
    t_rows = [dict(r, ok=(not r["s1_ok"]) if i % 2 == 0 else r["ok"])
              for i, r in enumerate(b_rows)]
    ns = {"items": items, "S1_ROWS": s1_rows, "B_ROWS": b_rows,
          "T_ROWS": t_rows, "b_skipped": 0, "t_skipped": 39,
          "json": json, "os": os,
          "norm_name": lambda s: re.sub(r"[^a-z0-9]+", "",
                                        (s or "").lower())}
    exec(compile(metrics_src.replace("/kaggle/working", tmp),
                 "metrics_cell", "exec"), ns)
    m = json.load(open(f"{tmp}/depth_metrics.json"))
    st = json.load(open(f"{tmp}/run_status.json"))
    assert m["b_n"] == 61 and m["b_emitted_n"] == 26
    assert m["t_n"] == 61 and m["paired_n"] == 61
    assert "t_s1_agree" in m and "t_vs_b_on_s1_wrong" in m
    for f in ("think_is_depth", "text_conditioning_helps",
              "conditioning_info_dead", "anchoring"):
        assert f in m
    assert st["arms"] == {"s1": "done", "b": "done", "t": "done"}
    assert os.path.exists(f"{tmp}/artifacts.zip")

