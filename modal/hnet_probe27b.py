"""Modal app: weakness diagnostics for qwen27b-int4-toolace vs bf16 base.

The general-quality gate (hnet_eval) found HumanEval pass@1 collapsed
-22.6pp under int4 while MMLU/ppl/toolace held. This app maps WHERE and
WHY:

  weight_map()   — per-tensor ||W_int4 - W_bf16|| / ||W_bf16|| for all 400
                   quantized linears, grouped by projection type and depth.
                   CPU-only, streams tensors (no full-model residency).

  probe_arm(arm) — generative probes with PER-ITEM dumps (the eval gate
                   only kept aggregates):
                     HumanEval all 164, greedy, extended failure classes
                       (syntax / assert / runtime / timeout / truncation)
                     GSM8K test N=100 4-shot CoT, #### extraction
                   arm in {"int4", "bf16"}; spawn both -> parallel H200s.

    modal deploy modal/hnet_probe27b.py
    modal run --detach modal/hnet_probe27b.py

Receipts -> /artifacts/{weight_error_27b,probe_{int4,bf16}}*.json
"""
import modal

app = modal.App("hnet-probe27b")
art = modal.Volume.from_name("hnet-artifacts", create_if_missing=True)
hfcache = modal.Volume.from_name("hnet-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "safetensors", "accelerate",
                 "huggingface_hub", "datasets", "numpy", "pyarrow", "pandas",
                 "zstandard", "flash-linear-attention")
    .add_local_dir("src", "/root/repo/src")
)

MODEL = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored"
INT4_DIR = "/artifacts/int4-27b-toolace"
HF_SNAP = ("/hf/hub/models--0bserverx--Qwen3.8-27B-Heretic-Abliterated-"
           "Uncensored/snapshots")
GSM8K_N = 100


# ---------------------------------------------------------------- weight map
@app.function(image=image, cpu=8, memory=131072, timeout=2 * 3600,
              volumes={"/artifacts": art, "/hf": hfcache})
def weight_map():
    import glob, json, os, re, sys
    sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    import torch
    from safetensors import safe_open
    from src.compress import dequant_int
    from src.int4_load import load_packed_state

    snaps = sorted(glob.glob(os.path.join(HF_SNAP, "*/")))
    assert snaps, f"no bf16 snapshot under {HF_SNAP}"
    snap = snaps[-1]
    index = json.load(open(os.path.join(snap, "model.safetensors.index.json")))
    wt2file = index["weight_map"]
    handles = {}

    def get_bf16(key):
        f = wt2file[key]
        if f not in handles:
            handles[f] = safe_open(os.path.join(snap, f), framework="pt",
                                   device="cpu")
        return handles[f].get_tensor(key)

    man = json.load(open(f"{INT4_DIR}/INT4_MANIFEST.json"))
    bits = man.get("bits", 4)
    bs = man.get("packing", {}).get("blocksize", 128)
    sd = load_packed_state(INT4_DIR)

    rows, missing = [], []
    for k, pk in sd.items():
        if not k.endswith(".qweight"):
            continue
        name = k[:-8]
        wkey = name + ".weight"
        if wkey not in wt2file:
            missing.append(name)
            continue
        wq = dequant_int(pk, sd[name + ".scales"].float(),
                         pk.shape[1] * 2, bits=bits, blocksize=bs).float()
        wb = get_bf16(wkey).float()
        d = wq - wb
        nw = wb.norm()
        rows.append({
            "name": name,
            "shape": list(wb.shape),
            "rel_l2": round(float(d.norm() / nw.clamp(min=1e-12)), 6),
            "cos": round(float((wq * wb).sum()
                               / (wq.norm() * nw).clamp(min=1e-12)), 6),
            "max_abs": round(float(d.abs().max()), 5),
            "w_std": round(float(wb.std()), 5),
        })
        del wq, wb, d
        if len(rows) % 50 == 0:
            print(f"[wmap] {len(rows)} tensors", flush=True)

    # aggregate by projection leaf-name and depth bucket
    def proj_of(n):
        return n.split(".")[-1]

    def layer_of(n):
        m = re.search(r"layers\.(\d+)", n)
        return int(m.group(1)) if m else -1

    import statistics as st
    by_proj, by_depth = {}, {}
    for r in rows:
        p, l = proj_of(r["name"]), layer_of(r["name"])
        by_proj.setdefault(p, []).append(r["rel_l2"])
        by_depth.setdefault(l // 8, []).append(r["rel_l2"])
    proj_tab = {p: {"n": len(v), "mean": round(st.mean(v), 5),
                    "max": round(max(v), 5)}
                for p, v in sorted(by_proj.items())}
    depth_tab = {str(b): {"layers": f"{b * 8}-{b * 8 + 7}",
                          "mean": round(st.mean(v), 5)}
                 for b, v in sorted(by_depth.items())}
    rels = sorted(r["rel_l2"] for r in rows)
    out = {
        "n_tensors": len(rows), "n_missing": len(missing),
        "missing": missing[:20],
        "rel_l2_mean": round(st.mean(rels), 6),
        "rel_l2_median": round(st.median(rels), 6),
        "rel_l2_p90": round(rels[int(0.9 * len(rels))], 6),
        "rel_l2_max": rels[-1],
        "by_proj": proj_tab,
        "by_depth8": depth_tab,
        "worst20": sorted(rows, key=lambda r: -r["rel_l2"])[:20],
        "best5": sorted(rows, key=lambda r: r["rel_l2"])[:5],
        "tensors": rows,
    }
    json.dump(out, open("/artifacts/weight_error_27b.json", "w"), indent=1)
    art.commit()
    print("RESULT_JSON " + json.dumps(
        {k: out[k] for k in ("n_tensors", "rel_l2_mean", "rel_l2_median",
                             "rel_l2_p90", "rel_l2_max", "by_proj")},
        default=str)[:4000], flush=True)
    print("[wmap] complete", flush=True)


# ------------------------------------------------------------------ probes
HE_CHECKER_V2 = r'''
import json, sys, signal, resource
def handler(s, f): raise TimeoutError()
signal.signal(signal.SIGALRM, handler)
items = json.load(open("/tmp/he_items.json"))
out = {}
for it in items:
    prog = it["prompt"] + it["completion"] + "\n" + it["test"] + \
        f"\ncheck({it['entry_point']})\n"
    try:
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        signal.alarm(15)
        ns = {}
        exec(compile(prog, "<he>", "exec"), ns)
        out[it["task_id"]] = {"pass": True}
    except SyntaxError:
        out[it["task_id"]] = {"pass": False, "kind": "syntax"}
    except TimeoutError:
        out[it["task_id"]] = {"pass": False, "kind": "timeout"}
    except AssertionError:
        out[it["task_id"]] = {"pass": False, "kind": "assert"}
    except Exception as e:
        out[it["task_id"]] = {"pass": False,
                              "kind": type(e).__name__}
    finally:
        signal.alarm(0)
json.dump(out, open("/tmp/he_results.json", "w"))
'''

GSM8K_FEWSHOT = """\
Q: Natalia sold clips to 48 of her friends in April, and then she sold \
half as many clips in May. How many clips did Natalia sell altogether in \
April and May?
A: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = \
<<48+24=72>>72 clips altogether in April and May.
#### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 \
minutes of babysitting. How much did she earn?
A: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, \
she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10

Q: Betty is saving money for a new wallet which costs $100. Betty has \
only half of the money she needs. Her parents decided to give her $15 \
for that purpose, and her grandparents twice as much as her parents. How \
much more money does Betty need to buy the wallet?
A: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's \
grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs \
100 - 50 - 15 - 30 = $<<100-50-15-30=5>>5 more. #### 5

Q: Julie is reading a 120-page book. Yesterday, she was able to read 12 \
pages and today, she read twice as many pages as yesterday. If she wants \
to read half of the remaining pages tomorrow, how many pages should she \
read?
A: Mary read 12 x 2 = <<12*2=24>>24 pages today. So she was able to read \
a total of 12 + 24 = <<12+24=36>>36 pages since yesterday. There are \
120 - 36 = <<120-36=84>>84 pages left to be read. Since she wants to \
read half of the remaining pages tomorrow, she should read 84 / 2 = \
<<84/2=42>>42 pages. #### 42

"""


def _load_gsm8k():
    from datasets import load_dataset
    for repo, cfg in (("openai/gsm8k", "main"),
                      ("yentinglin/gsm8k_main", None),
                      ("iknowchinatravel/gsm8k", None)):
        try:
            if cfg:
                return load_dataset(repo, cfg, split="test")
            return load_dataset(repo, split="test")
        except Exception as e:
            print(f"[gsm8k] {repo} failed: {type(e).__name__} {e}",
                  flush=True)
    raise RuntimeError("no gsm8k source worked")


def _norm_num(s):
    import re
    s = s.strip().replace(",", "").rstrip(".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = m.group(0)
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


@app.function(image=image, gpu="H200", timeout=4 * 3600, memory=196608,
              volumes={"/artifacts": art, "/hf": hfcache},
              secrets=[modal.Secret.from_name("hf-token")])
def probe_arm(arm: str):
    import os, sys, json, time, gc, random, subprocess, signal, resource
    import re as _re
    os.environ["HF_HOME"] = "/hf"
    sys.path.insert(0, "/root/repo")
    import torch
    from src.compress import dequant_int
    from src.int4_load import load_packed_state
    from datasets import load_dataset

    out = {"arm": arm}

    # ---------------- model load (same paths as hnet_eval) --------------
    def load_int4_model():
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
        tok = AutoTokenizer.from_pretrained(INT4_DIR)
        return model, tok

    def load_bf16_model():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="auto",
            max_memory={0: "130GiB", "cpu": "80GiB"})
        model.eval()
        tok = AutoTokenizer.from_pretrained(MODEL)
        return model, tok

    loader = load_int4_model if arm == "int4" else load_bf16_model
    t0 = time.time()
    model, tok = loader()
    print(f"[{arm}] loaded ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- HumanEval, all 164, per-item ----------------------
    @torch.no_grad()
    def eval_he():
        ds = load_dataset("openai/openai_humaneval",
                          "openai_humaneval", split="test")
        items, t0 = [], time.time()
        for i, row in enumerate(ds):
            pids = tok(row["prompt"], return_tensors="pt").input_ids.cuda()
            gen = model.generate(
                pids, max_new_tokens=256, do_sample=False,
                pad_token_id=tok.eos_token_id)
            comp = tok.decode(gen[0][pids.shape[1]:],
                              skip_special_tokens=True)
            items.append({"task_id": row["task_id"],
                          "prompt": row["prompt"], "completion": comp,
                          "test": row["test"],
                          "entry_point": row["entry_point"],
                          "trunc": bool(gen.shape[1]
                                        >= pids.shape[1] + 256)})
            if (i + 1) % 32 == 0:
                print(f"[{arm}] he {i + 1}/164 ({time.time() - t0:.0f}s)",
                      flush=True)
        open("/tmp/he_items.json", "w").write(json.dumps(items))
        open("/tmp/he_check.py", "w").write(HE_CHECKER_V2)
        r = subprocess.run([sys.executable, "/tmp/he_check.py"],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(f"[{arm}] checker stderr: {r.stderr[-1500:]}", flush=True)
        res = json.load(open("/tmp/he_results.json"))
        npass = sum(1 for v in res.values() if v["pass"])
        kinds = {}
        for v in res.values():
            if not v["pass"]:
                kinds[v.get("kind", "?")] = kinds.get(
                    v.get("kind", "?"), 0) + 1
        return {"pass": npass, "n": len(res),
                "acc": round(npass / max(len(res), 1), 4),
                "fail_kinds": kinds,
                "items": items, "results": res}

    # ---------------- GSM8K, N=100, 4-shot ------------------------------
    @torch.no_grad()
    def eval_gsm8k():
        ds = _load_gsm8k()
        n = min(GSM8K_N, len(ds))
        idx = sorted(random.Random(0).sample(range(len(ds)), n))
        items, t0 = [], time.time()
        for j, i in enumerate(idx):
            row = ds[i]
            q, gold = row["question"], row["answer"].split("####")[-1].strip()
            prompt = (GSM8K_FEWSHOT + f"Q: {q}\nA:")
            pids = tok(prompt, return_tensors="pt").input_ids.cuda()
            gen = model.generate(
                pids, max_new_tokens=512, do_sample=False,
                pad_token_id=tok.eos_token_id)
            comp = tok.decode(gen[0][pids.shape[1]:],
                              skip_special_tokens=True)
            if "####" in comp:
                pred = _norm_num(comp.split("####")[-1])
            else:
                nums = _re.findall(r"-?\d[\d,]*(?:\.\d+)?", comp)
                pred = _norm_num(nums[-1]) if nums else None
            g = _norm_num(gold)
            items.append({"idx": int(i), "question": q, "gold": gold,
                          "pred": pred, "correct": pred == g,
                          "completion": comp[:3000],
                          "trunc": bool(gen.shape[1]
                                        >= pids.shape[1] + 512)})
            if (j + 1) % 20 == 0:
                print(f"[{arm}] gsm8k {j + 1}/{n} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        nc = sum(1 for it in items if it["correct"])
        return {"right": nc, "n": n, "acc": round(nc / max(n, 1), 4),
                "items": items}

    for name, fn in (("humaneval", eval_he), ("gsm8k", eval_gsm8k)):
        try:
            out[name] = fn()
        except Exception as e:
            import traceback
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        json.dump(out, open(f"/artifacts/probe_{arm}.json", "w"), indent=1)
        art.commit()
        print(f"[{arm}] {name} -> "
              f"{ {k: v for k, v in out[name].items() if k != 'items'} }",
              flush=True)

    print("RESULT_JSON " + json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "items"}
         for k, v in out.items() if isinstance(v, dict)}), flush=True)
    print(f"[{arm}] complete", flush=True)


@app.local_entrypoint()
def main():
    c0 = weight_map.spawn()
    c1 = probe_arm.spawn("int4")
    c2 = probe_arm.spawn("bf16")
    print(f"spawned wmap={c0.object_id} int4={c1.object_id} "
          f"bf16={c2.object_id}", flush=True)
