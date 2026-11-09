"""Modal app: GENERAL-QUALITY eval gate for the packed-int4 27B trunk.

Question this answers: is qwen27b-int4-toolace actually a good model, not
just good at tool-name selection? Runs the SAME battery on (a) the int4
artifact dequantized-on-load and (b) the fresh bf16 base, so every delta is
attributable to compression vs capability.

Battery:
  MMLU 5-shot (cais/mmlu 'all', 57 subj x <=20 test, dev as exemplars)
  HumanEval pass@1 greedy (openai/openai_humaneval, 164 problems, exec-checked)
  WikiText-2 perplexity (~16k tokens, 2048-window teacher forcing)
  pinned 40-item ToolACE name exam (receipt continuity)

Per AGENTS.md: deploy + spawn detached, results to Modal Volume + RESULT_JSON.
    modal deploy modal/hnet_eval.py   # or modal run --detach
"""
import modal

app = modal.App("hnet-eval")
art = modal.Volume.from_name("hnet-artifacts", create_if_missing=True)
hfcache = modal.Volume.from_name("hnet-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "safetensors", "accelerate",
                 "huggingface_hub", "datasets", "numpy", "pyarrow", "pandas",
                 "flash-linear-attention")
    .add_local_dir("src", "/root/repo/src")
)

MODEL = "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored"
INT4_DIR = "/artifacts/int4-27b-toolace"
MMLU_PER_SUBJ = int(__import__("os").environ.get("MMLU_PER_SUBJ", "20"))
HE_MAX_NEW = int(__import__("os").environ.get("HE_MAX_NEW", "256"))
PPL_TOKENS = int(__import__("os").environ.get("PPL_TOKENS", "16384"))


@app.function(image=image, gpu="H200", timeout=6 * 3600, memory=196608,
              volumes={"/artifacts": art, "/hf": hfcache},
              secrets=[modal.Secret.from_name("hf-token")])
def run_eval():
    import os, sys, json, time, gc, random, subprocess, signal, resource
    os.environ["HF_HOME"] = "/hf"
    sys.path.insert(0, "/root/repo")
    import torch
    from src.compress import dequant_int
    from src.int4_load import load_packed_state
    from src.probe_common import stream_toolace

    receipts = {}

    # ---------- loaders -------------------------------------------------
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
        missing, unexpected = model.load_state_dict(full, strict=False)
        real_missing = [k for k in missing if not k.endswith(".bias")]
        print(f"[load] int4 missing={real_missing[:5]} "
              f"unexpected={unexpected[:5]}", flush=True)
        del full
        gc.collect()
        model = model.to("cuda")
        model.eval()
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

    # ---------- evals ----------------------------------------------------
    @torch.no_grad()
    def eval_mmlu(model, tok):
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all")
        test, dev = ds["test"], ds["dev"]
        dev_by_subj, test_by_subj = {}, {}
        for r in dev:
            dev_by_subj.setdefault(r["subject"], []).append(r)
        for r in test:
            test_by_subj.setdefault(r["subject"], []).append(r)
        rng = random.Random(0)
        items = []
        for subj in sorted(test_by_subj):
            rows = test_by_subj[subj]
            k = min(MMLU_PER_SUBJ, len(rows))
            idx = sorted(rng.sample(range(len(rows)), k))
            ex = dev_by_subj.get(subj, [])[:5]
            for i in idx:
                items.append((subj, ex, rows[i]))
        letters = "ABCD"
        cand = []
        for L in letters:
            for pre in (" " + L, L):
                ids = tok.encode(pre, add_special_tokens=False)
                if len(ids) == 1:
                    cand.append(ids[0]); break
        assert len(cand) == 4, f"non-single-token answers: {cand}"

        def fmt_choices(ch):
            return "\n".join(f"{letters[i]}. {c}" for i, c in
                             enumerate(ch))
        head = ("The following are multiple choice questions (with "
                "answers) about {s}.\n\n")
        right = n = 0
        per_subj = {}
        t0 = time.time()
        for subj, exs, row in items:
            parts = [head.format(s=subj.replace("_", " "))]
            for e in exs:
                gold = letters[int(e["answer"])]
                parts.append(
                    f"Q: {e['question']}\n{fmt_choices(e['choices'])}\n"
                    f"Answer: {gold}\n\n")
            parts.append(
                f"Q: {row['question']}\n{fmt_choices(row['choices'])}\n"
                f"Answer:")
            ids = tok("".join(parts), return_tensors="pt").input_ids[0]
            logits = model(ids.unsqueeze(0).cuda()).logits[0, -1].float()
            pred = letters[int(torch.argmax(logits[cand]))]
            ok = pred == letters[int(row["answer"])]
            right += ok; n += 1
            a, b = per_subj.get(subj, (0, 0))
            per_subj[subj] = (a + ok, b + 1)
        acc = right / max(n, 1)
        worst = sorted(per_subj.items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]
        print(f"[mmlu] {right}/{n} acc={acc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return {"right": right, "n": n, "acc": round(acc, 4),
                "worst_subj": {k: f"{v[0]}/{v[1]}" for k, v in worst}}

    # ---- humaneval ----
    HE_CHECKER = r'''
import json, sys, signal, resource, traceback
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
        out[it["task_id"]] = True
    except Exception:
        out[it["task_id"]] = False
    finally:
        signal.alarm(0)
json.dump(out, open("/tmp/he_results.json", "w"))
'''

    def eval_humaneval(model, tok):
        from datasets import load_dataset
        try:
            ds = load_dataset("openai/openai_humaneval",
                              "openai_humaneval", split="test")
        except Exception:
            ds = load_dataset("openai/openai_humaneval", split="test")
        items = []
        t0 = time.time()
        for i, row in enumerate(ds):
            pids = tok(row["prompt"], return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                gen = model.generate(
                    pids, max_new_tokens=HE_MAX_NEW, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            comp = tok.decode(gen[0][pids.shape[1]:],
                              skip_special_tokens=True)
            items.append({"task_id": row["task_id"],
                          "prompt": row["prompt"], "completion": comp,
                          "test": row["test"],
                          "entry_point": row["entry_point"]})
            if (i + 1) % 32 == 0:
                json.dump(items, open("/tmp/he_items.json", "w"))
                print(f"[humaneval] generated {i + 1}/164 "
                      f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(items, open("/tmp/he_items.json", "w"))
        open("/tmp/he_check.py", "w").write(HE_CHECKER)
        r = subprocess.run([sys.executable, "/tmp/he_check.py"],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print("[humaneval] checker stderr:", r.stderr[-2000:],
                  flush=True)
        res = json.load(open("/tmp/he_results.json"))
        npass = sum(res.values())
        print(f"[humaneval] pass@1 {npass}/164 "
              f"({time.time() - t0:.0f}s)", flush=True)
        return {"pass": npass, "n": len(res),
                "acc": round(npass / max(len(res), 1), 4)}

    @torch.no_grad()
    def eval_ppl(model, tok):
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="test")
        text = "\n\n".join(r["text"] for r in ds
                           if r["text"].strip())[:600000]
        ids = tok(text, return_tensors="pt").input_ids[0][:PPL_TOKENS]
        W = 2048
        nll, cnt = 0.0, 0
        for s in range(0, len(ids) - 1, W):
            w = ids[s:s + W + 1]
            if len(w) < 3:
                break
            logits = model(w.unsqueeze(0).cuda()).logits[0, :-1].float()
            tgt = w[1:].cuda()
            nll += torch.nn.functional.cross_entropy(
                logits, tgt, reduction="sum").item()
            cnt += len(tgt)
        import math
        return {"nll": round(nll / cnt, 4), "ppl": round(math.exp(nll / cnt), 3),
                "tokens": cnt}

    @torch.no_grad()
    def eval_toolace(model, tok):
        def prefix_lp(ctx, cand):
            x = torch.tensor([ctx + cand], device="cuda")
            logits = model(x).logits[0]
            s = len(ctx) - 1
            lp = torch.log_softmax(logits[s:s + len(cand)].float(), -1)
            return float(sum(lp[i, t] for i, t in enumerate(cand)))
        right = none_n = decl = decl_right = 0
        for it in stream_toolace(2500, 40):
            ctx = tok(f"{it['sys']}\nUSER: {it['user']}\nASSISTANT: [",
                      return_tensors="pt").input_ids[0].tolist()
            sc = {n: prefix_lp(ctx, tok(f"{n}(", add_special_tokens=False)
                               ["input_ids"])
                  for n in it["tools"] + ["none"]}
            pred = max(sc, key=sc.get)
            declared = it["gold"] in it["tools"]
            decl += declared
            decl_right += declared and pred == it["gold"]
            right += pred == it["gold"]
            none_n += pred == "none"
        return {"right": right, "declared": decl, "decl_right": decl_right,
                "none": none_n, "n": 40}

    # ---------- run both models -----------------------------------------
    for tag, loader in (("int4", load_int4_model), ("bf16", load_bf16_model)):
        print(f"[eval] === loading {tag} ===", flush=True)
        t0 = time.time()
        model, tok = loader()
        print(f"[eval] {tag} loaded ({time.time() - t0:.0f}s)", flush=True)
        rec = {}
        for name, fn in (("mmlu", eval_mmlu), ("humaneval", eval_humaneval),
                         ("ppl", eval_ppl), ("toolace40", eval_toolace)):
            try:
                rec[name] = fn(model, tok)
            except Exception as e:
                import traceback
                rec[name] = {"error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
            receipts[tag] = rec
            json.dump(receipts, open("/artifacts/eval_receipts.json", "w"),
                      indent=1)
            art.commit()
            print(f"[eval] {tag}.{name} -> {rec[name]}", flush=True)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    json.dump(receipts, open("/artifacts/eval_receipts.json", "w"), indent=1)
    art.commit()
    print("RESULT_JSON " + json.dumps(receipts), flush=True)
    print("[eval] complete", flush=True)


@app.local_entrypoint()
def main():
    call = run_eval.spawn()
    print(f"spawned call_id={call.object_id}", flush=True)
