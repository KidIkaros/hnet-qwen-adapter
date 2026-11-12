#!/usr/bin/env python3
"""Build the Kaggle TwIL-8B leg-4 kernel (2xT4, ~5.5h, two arms).

Per docs/SPEC_TWIL_LEG3.md (leg-4 addendum):
  Leg 0  load frozen TwIL-8B (device-map-safe shim, proven v3 pattern)
  Leg 1  v0 sum-embedder baseline bpb on REAL held-out eval docs
  Leg 2  train LearnedPatchEmbedder E[256,period,D] (warm-start = v0
         construction), two arms: period-2 @3000 + period-1 @1500 steps,
         DOC_CAP_TRAIN=512, bpb smoke gate, learning-curve evals,
         effective_bpb sanity, wall-clock guard
  Leg 3  learned-embedder eval on same docs -> delta vs baseline
  Leg 4  artifacts.zip (adapter ckpt + eval JSONs + corpus manifest)

Corpus: deployment-matched agentic mix (owner-directed) — tool-use 0.40
(nemotron_terminal/open_swe_traces/ToolACE), code 0.25, reasoning 0.20,
prose 0.15. Eval docs non-repeated, per-domain, disjoint stream positions.
"""
import json

OUT = "/tmp/twil_leg4.ipynb"
REPO = "/home/ikaaros/Coding/Gold/hnet-qwen-adapter"
SRC_FILES = ["src/__init__.py", "src/adapter.py", "src/corpus.py"]


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in text.split("\n")]}


def code_cell(text):
    return {"cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in text.split("\n")]}


def src_writer_cell():
    import base64, os
    parts = ["import os, base64\n",
             "os.makedirs('/kaggle/working/src', exist_ok=True)\n"]
    for rel in SRC_FILES:
        content = open(os.path.join(REPO, rel), "rb").read()
        b64 = base64.b64encode(content).decode()
        parts.append(
            f"open('/kaggle/working/src/{os.path.basename(rel)}','wb')"
            f".write(base64.b64decode('{b64}'))\n")
    parts.append("import sys; sys.path.insert(0, '/kaggle/working')\n")
    parts.append("from src.adapter import (PeriodicChunker, PatchEmbedder, "
                 "ByteHead, effective_bpb)\n")
    parts.append("from src.corpus import byte_token_map\n")
    parts.append("print('src bundle OK')\n")
    return code_cell("".join(parts))


cells = [
    md_cell("""# TwIL-8B leg-4 — p1 NaN fix + Arm A resume to 6000 (2xT4)

Per `docs/SPEC_TWIL_LEG3.md` (leg-4 addendum). Trains a learned patch embedder
(`E[256, period, D]`, warm-started from the v0 sum construction) on
frozen TwIL-8B hidden states; measures first-byte bpb delta on real,
non-repeated, deployment-matched eval docs. The existential measurement
for Route A."""),

    code_cell("""# Cell 1: deps — transformers pinned to the version twil-probe proved
import subprocess, sys, time
print('installing transformers==5.20.2 datasets...', flush=True)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'transformers==5.20.2', 'datasets',
                    'huggingface_hub'],
                   capture_output=True, text=True)
print(r.stdout[-400:], r.stderr[-400:], flush=True)
import torch
assert torch.cuda.is_available(), 'GPU required'
ngpu = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(ngpu)]
T_START = time.time()   # wall-clock guard anchor (GPU sessions bill idle)
print('GPUs:', ngpu, names, flush=True)
assert ngpu >= 2, f'need 2xT4, got {ngpu}'
"""),

    src_writer_cell(),

    code_cell("""# Cell 3: load frozen TwIL-8B + AdapterShim (eval + train paths)
# v3-proven pattern: from_pretrained(REPO_ID) -> HF cache on ROOT disk.
# Do NOT snapshot_download into /kaggle/working — 16.4GB vs 19.5GB quota
# + xet reconstruction overhead = disk-full (leg1 v1 postmortem).
import torch, torch.nn.functional as F, numpy as np, json, time, os, math
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.adapter import PatchEmbedder, ByteHead, PeriodicChunker, effective_bpb
from src.corpus import byte_token_map

os.system('df -h / /kaggle/working /tmp 2>/dev/null | head -6')
REPO_ID = 'webAI-Official/TwIL-8B'
tok = AutoTokenizer.from_pretrained(REPO_ID)
model = AutoModelForCausalLM.from_pretrained(
    REPO_ID, torch_dtype=torch.bfloat16, device_map='auto',
    max_memory={0: '14GiB', 1: '14GiB'})
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
# grad checkpointing: backward needs activations; recompute-on-backward
# bounds memory on T4 (reentrant=False required for inputs_embeds grad flow)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={'use_reentrant': False})
print('model loaded + checkpointing enabled', flush=True)

PERIOD = 2
DOC_CAP_TRAIN, DOC_CAP_EVAL = 512, 4096
bmap = byte_token_map(tok)
print(f'byte coverage: {len(bmap)}/256', flush=True)

class AdapterShim:
    \"\"\"Device-map-safe ChunkLM with pluggable embedder + a training path.
    Never .to() the split model; forwards go through accelerate hooks.
    No attention_mask: q_len==kv_len, no padding -> create_causal_mask
    returns None -> sdpa is_causal fast path.\"\"\"
    def __init__(self, model, tok, period=PERIOD, lr=3e-4, clip=1.0):
        self.model = model
        self.tok = tok
        self.chunker = PeriodicChunker(period)
        self.v0 = PatchEmbedder(model, tok)          # frozen sum construction
        self.byte_head = ByteHead(tok)
        self.period = period
        self.dtype = model.dtype
        self.bmap = byte_token_map(tok)
        self.mapped = set(self.bmap.keys())
        # learned table E[256, period, D] fp32 on cuda:0
        emb = model.get_input_embeddings().weight
        D = emb.shape[1]
        E = torch.zeros(256, period, D, dtype=torch.float32)
        with torch.no_grad():
            for b, tid in self.bmap.items():
                E[b, :, :] = emb[tid].float()        # warm start == v0 sum
        self.E = nn.Parameter(E.to('cuda:0'))
        self.clip = clip
        self.opt = torch.optim.AdamW([self.E], lr=lr)
        self.nonfinite = False   # leg-3 v1: p1 E went NaN in 50 steps

    def embed(self, spans, mode):
        if mode == 'v0':
            pv, _ = self.v0.embed_all([b for _, b in spans])
            return pv                                 # (M, D) fp32, no grad
        flat_b, flat_p, sidx = [], [], []
        for si, (_, chunk) in enumerate(spans):
            for p, b in enumerate(chunk):
                flat_b.append(b)
                flat_p.append(min(p, self.period - 1))
                sidx.append(si)
        bt = torch.tensor(flat_b, device='cuda:0')
        pt = torch.tensor(flat_p, device='cuda:0')
        st = torch.tensor(sidx, device='cuda:0')
        rows = self.E[bt, pt]                         # (N, D) grad
        M = len(spans)
        v = torch.zeros(M, self.E.shape[2], device='cuda:0')
        v.index_add_(0, st, rows)
        return v

    def _forward(self, pv, grad):
        x = pv.unsqueeze(0).to('cuda:0', self.dtype)
        if grad:
            base = self.model.model(inputs_embeds=x)
        else:
            with torch.no_grad():
                base = self.model.model(inputs_embeds=x)
        return base.last_hidden_state[0]              # (M, D) model dtype

    def _lm_rows(self, hs, rows):
        # full-vocab logits on selected positions only — bounds the (K,V)
        # peak.  hs fp32 -> cast to head dtype at the boundary or F.linear
        # raises float != BFloat16.  no_grad: eval readout never carries grad.
        lm = self.model.lm_head
        dev = lm.weight.device
        idx = (torch.as_tensor(rows, device=hs.device) if rows is not None
               else torch.arange(hs.shape[0], device=hs.device))
        with torch.no_grad():
            return lm(hs[idx].to(dev, self.dtype))

    def _byte_logits(self, hs, tpos):
        # TRAIN readout: byte-restricted logits hs[tpos] @ W_byte.T — gradient
        # flows into hs (-> trunk -> embedder); W rows are frozen constants.
        # Eval uses _lm_rows + byte_probs instead (canonical contract).
        lm = self.model.lm_head
        bids = sorted(self.bmap)
        W = lm.weight[[self.bmap[b] for b in bids]]
        sel = hs[tpos].to(lm.weight.device, self.dtype)
        return sel @ W.T, bids                        # (K, n_bytes)

    def _targets(self, data, spans):
        tpos, tbytes = [], []
        for m in range(len(spans) - 1):
            t = spans[m + 1][0]
            b = int(data[t])
            if b in self.mapped:
                tpos.append(m); tbytes.append(b)
        return tpos, tbytes

    def boundary_nll(self, data, mode='v0'):
        # (pos, nll_bits, cov, n_spans) — next-chunk-first-byte predictions
        # on mapped targets, canonical byte_probs readout.
        spans = self.chunker.spans(data)
        if len(spans) < 2:
            raise ValueError(f'too few chunks: {len(spans)}')
        tpos, tbytes = self._targets(data, spans)
        if not tpos:
            raise ValueError('no mapped target bytes')
        pv = self.embed(spans, mode)
        hs = self._forward(pv, grad=False)
        logits = self._lm_rows(hs, tpos)                # (K, V) model dtype
        pb = self.byte_head.byte_probs(logits)        # (K, 256) byte marginal
        tb = torch.tensor(tbytes, device=pb.device)
        nll = -torch.log2(pb.gather(1, tb[:, None]).clamp_min(1e-12)) \\
            .double().cpu().numpy().ravel()
        pos = np.array([spans[m + 1][0] for m in range(len(spans) - 1)])
        if mode == 'learned':
            cov = pv.abs().sum(1).gt(0).float().mean().item()
        else:
            cov = len([b for _, ch in spans for b in ch if b in self.mapped]) \\
                / max(sum(len(c) for _, c in spans), 1)
        return pos, nll, cov, len(spans)

    def first_byte_bpb(self, data, mode='v0'):
        # canonical contract (mirrors ChunkLM.first_byte_bpb / v3 shim):
        # FULL-vocab logits -> ByteHead.byte_probs marginalization.  Using
        # the same readout for both modes keeps the delta unconfounded.
        pos, nll, cov, nspans = self.boundary_nll(data, mode)
        tcov = len(nll) / max(nspans - 1, 1)
        return float(nll.sum() / len(nll)), len(nll), (cov, tcov)

    def byte_nll(self, data, mode='v0'):
        # period-1 pass for effective_bpb: every byte is a chunk position.
        # Streams lm_head in 2048-row blocks so (T,V) logits never materialize.
        spans = PeriodicChunker(1).spans(data)
        pv = self.embed(spans, mode)
        hs = self._forward(pv, grad=False)            # (T, D)
        nll = np.full(len(data), 8.0)
        tgt = None
        B = 2048
        for i in range(0, len(data) - 1, B):
            j = min(i + B, len(data) - 1)
            logits = self._lm_rows(hs, list(range(i, j)))   # (j-i, V)
            pb = self.byte_head.byte_probs(logits)
            if tgt is None:
                tgt = torch.tensor(list(data[1:]), device=pb.device)
            nll[i + 1:j + 1] = -torch.log2(
                pb.gather(1, tgt[i:j, None]).clamp_min(1e-12)) \\
                .double().cpu().numpy().ravel()
        return nll

    def train_step(self, data):
        spans = self.chunker.spans(data)
        if len(spans) < 2:
            return None
        tpos, tbytes = self._targets(data, spans)
        if len(tpos) < 8:
            return None
        self.opt.zero_grad(set_to_none=True)
        pv = self.embed(spans, 'learned')
        hs = self._forward(pv, grad=True)
        logits, bids = self._byte_logits(hs, tpos)
        bidx = {b: i for i, b in enumerate(bids)}
        tgt = torch.tensor([bidx[b] for b in tbytes], device=logits.device)
        loss = F.cross_entropy(logits.float(), tgt) / math.log(2)   # bits
        if not torch.isfinite(loss):
            # leg-3 lesson: a poisoned doc's NaN loss poisons E via grads
            self.opt.zero_grad(set_to_none=True)
            return None
        loss.backward()
        # leg-3 v1 lesson: period-1 gradients explode without clipping
        torch.nn.utils.clip_grad_norm_([self.E], self.clip)
        self.opt.step()
        if not torch.isfinite(self.E).all():
            self.nonfinite = True
        return float(loss.detach())

shim = AdapterShim(model, tok)
print('AdapterShim ready (period', PERIOD, ')', flush=True)
"""),

    code_cell("""# Cell 4: corpus — deployment-matched agentic mix, train/eval disjoint
# Single contiguous pass per source: first q*cap bytes -> train docs,
# next 4*cap bytes -> eval docs. Disjoint BY CONSTRUCTION (same stream,
# non-overlapping slices). Buffer-shuffle before the take spreads the
# sample across shards (RESEARCH_DATA_ORDERING head-of-stream mitigation).
import zlib
from datasets import load_dataset

DOLMA3_REPO = 'allenai/dolma3_mix-6T'
SOURCES = {
    'nemotron_terminal': ('nvidia/Nemotron-Terminal-Corpus', None,
                          'skill_based_mixed', 'train'),
    'open_swe_traces': ('nvidia/Open-SWE-Traces', None, 'v2.0', 'claudecode'),
    'stack_edu': (DOLMA3_REPO, 'data/stack_edu-*/*.jsonl.zst', None, 'train'),
    'finemath_3plus': (DOLMA3_REPO, 'data/finemath-3plus/*.jsonl.zst',
                       None, 'train'),
    'proofpile_arxiv': (DOLMA3_REPO, 'data/rpj-proofpile-arxiv/*.jsonl.zst',
                        None, 'train'),
    'common_crawl': (DOLMA3_REPO, 'data/common_crawl-*/*.jsonl.zst',
                     None, 'train'),
    'dolma_wiki': (DOLMA3_REPO, 'data/dolma1_7-wiki-en/*.jsonl.zst',
                   None, 'train'),
}
MIX = {'nemotron_terminal': 0.15, 'open_swe_traces': 0.15, 'toolace': 0.10,
       'stack_edu': 0.25, 'finemath_3plus': 0.12, 'proofpile_arxiv': 0.08,
       'common_crawl': 0.10, 'dolma_wiki': 0.05}
N_TRAIN_TOTAL, N_EVAL_PER = 1600, 4

def _text_of(row):
    t = row.get('text')
    if isinstance(t, str) and len(t) > 200:
        return t
    for k in ('conversations', 'messages', 'trajectory', 'content'):
        v = row.get(k)
        if isinstance(v, list) and v:
            parts = []
            for m in v:
                if isinstance(m, dict):
                    parts.append(str(m.get('content') or m.get('value') or ''))
                else:
                    parts.append(str(m))
            s = '\\n'.join(p for p in parts if p)
            if len(s) > 200:
                return s
    best = ''
    for v in row.values():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best if len(best) > 200 else json.dumps(row, default=str)

def stream_source(key):
    repo, glob, cfg, split = SOURCES[key]
    kw = {'split': split, 'streaming': True}
    if glob:
        kw['data_files'] = glob
    ds = load_dataset(repo, cfg, **kw) if cfg else load_dataset(repo, **kw)
    return ds.shuffle(seed=zlib.crc32(key.encode()), buffer_size=256)

def take_docs(key, n_train, n_eval):
    TRAIN_REGION = n_train * 2048   # leg-1 byte layout anchor (eval offset)
    need = TRAIN_REGION + n_eval * DOC_CAP_EVAL
    ds = stream_source(key)
    buf = b''
    for row in ds:
        buf += _text_of(row).encode('utf-8', 'ignore')
        if len(buf) >= need:
            break
    tr = [buf[i:i + DOC_CAP_TRAIN]
          for i in range(0, TRAIN_REGION, DOC_CAP_TRAIN)]
    e0 = TRAIN_REGION
    ev = [buf[e0 + i:e0 + i + DOC_CAP_EVAL]
          for i in range(0, n_eval * DOC_CAP_EVAL, DOC_CAP_EVAL)]
    return tr, ev

train_docs, eval_docs, manifest = [], [], {}
for key, w in MIX.items():
    q = max(1, round(N_TRAIN_TOTAL * w / sum(MIX.values())))
    if key == 'toolace':
        ds = load_dataset('Team-ACE/ToolACE', split='train', streaming=True)
        rows = []
        for i, r in enumerate(ds):
            if i < 5000:
                continue
            conv = r.get('conversations') or []
            txt = '\\n\\n'.join(str(c.get('value') or '') for c in conv
                               if c.get('value'))
            if len(txt) > 400:
                rows.append(txt.encode('utf-8', 'ignore'))
            if len(rows) >= q + N_EVAL_PER:
                break
        tr = []
        for b in rows[:q]:
            for i in range(0, len(b), DOC_CAP_TRAIN):
                tr.append(b[i:i + DOC_CAP_TRAIN])
            if len(tr) >= 4 * q:
                break
        tr = tr[:4 * q]   # ~q*2048 bytes = leg-1 train budget
        ev = [b[:DOC_CAP_EVAL] for b in rows[q:q + N_EVAL_PER]]
        manifest['toolace'] = {'train': len(tr), 'eval': len(ev),
                               'rows_from': 5000}
    else:
        try:
            tr, ev = take_docs(key, q, N_EVAL_PER)
            manifest[key] = {'train': len(tr), 'eval': len(ev)}
        except Exception as e:
            # per-source isolation: one bad schema must not kill the mix
            manifest[key] = {'error': f'{type(e).__name__}: {str(e)[:150]}'}
            tr, ev = [], []
            print(f'[mix] {key}: FAILED {type(e).__name__} — skipped',
                  flush=True)
    train_docs += tr
    eval_docs += [(key, b) for b in ev]
    print(f'[mix] {key}: train={len(tr)} eval={len(ev)}', flush=True)

import random
random.Random(0).shuffle(train_docs)
json.dump({'mix': MIX, 'manifest': manifest,
           'n_train': len(train_docs), 'n_eval': len(eval_docs),
           'caps': {'train': DOC_CAP_TRAIN, 'eval': DOC_CAP_EVAL}},
          open('/kaggle/working/corpus_manifest.json', 'w'), indent=1)
print(f'[corpus] {len(train_docs)} train + {len(eval_docs)} eval docs',
      flush=True)
"""),

    code_cell("""# Cell 5: Leg 1 — v0 baseline bpb on real eval docs (per-domain)
base_results = []
t0 = time.time()
for k, (dom, doc) in enumerate(eval_docs):
    try:
        bpb, n, (icov, tcov) = shim.first_byte_bpb(doc, mode='v0')
        base_results.append({'domain': dom, 'bpb': bpb, 'n_preds': n,
                             'input_cov': icov, 'target_cov': tcov})
    except Exception as e:
        base_results.append({'domain': dom,
                             'error': f'{type(e).__name__}: {str(e)[:150]}'})
    if k % 8 == 0:
        print(f'  baseline {k}/{len(eval_docs)} ({time.time()-t0:.0f}s)',
              flush=True)
okb = [r for r in base_results if 'error' not in r]
by_dom = {}
for r in okb:
    by_dom.setdefault(r['domain'], []).append(r['bpb'])
base_summary = {d: round(sum(v)/len(v), 4) for d, v in by_dom.items()}
base_overall = round(sum(r['bpb'] for r in okb) / max(len(okb), 1), 4)
print(f'[leg4-baseline] overall first_byte_bpb={base_overall} '
      f'per-domain={base_summary}', flush=True)
# fixed curve subset (shared by both arms) + period-1 v0 baseline
# over ALL eval docs (spec: period comparison on the same eval docs)
curve_docs = eval_docs[::max(1, len(eval_docs)//8)][:8]
shim_p1 = AdapterShim(model, tok, period=1)
base_p1 = []
for dom, doc in eval_docs:
    try:
        bpb, n, _ = shim_p1.first_byte_bpb(doc, mode='v0')
        base_p1.append({'domain': dom, 'bpb': bpb, 'n_preds': n})
    except Exception as e:
        base_p1.append({'domain': dom,
                        'error': f'{type(e).__name__}: {str(e)[:120]}'})
okp1 = [r for r in base_p1 if 'error' not in r]
base_p1_overall = round(sum(r['bpb'] for r in okp1) / max(len(okp1), 1), 4)
print(f'[leg4-baseline-p1] v0 period-1 curve-subset bpb={base_p1_overall}',
      flush=True)
json.dump({'results': base_results, 'summary': base_summary,
           'overall': base_overall},
          open('/kaggle/working/leg4_baseline.json', 'w'), indent=1)
json.dump({'results': base_p1, 'overall': base_p1_overall},
          open('/kaggle/working/leg4_baseline_p1.json', 'w'), indent=1)
"""),

    code_cell("""# Cell 6: train_arm() + Arm B FIRST (period-1, NaN fixes)
# leg-3 v3: p1 E went non-finite at step 44 (flat loss, no learning).
# Fixes: lr 3e-4 -> 1e-4, clip 1.0 -> 0.5, finite-loss pre-check in
# train_step (poisoned docs skipped before backward).
def train_arm(shim_a, target_ok, curve_at, ckpt_path, tag, gate=True):
    save = lambda: torch.save(
        {'E': shim_a.E.detach().cpu(), 'period': shim_a.period}, ckpt_path)
    curve = []
    def curve_eval(step_tag):
        tot_nll = tot_n = ndone = 0
        for dom, doc in curve_docs:
            try:
                torch.cuda.empty_cache()
                bpb, n, _ = shim_a.first_byte_bpb(doc, mode='learned')
                tot_nll += bpb * n; tot_n += n; ndone += 1
            except Exception as e:
                print(f'  {tag} curve[{step_tag}] {dom} err '
                      f'{type(e).__name__}', flush=True)
        if tot_n:
            curve.append({'step': step_tag, 'bpb': tot_nll / tot_n,
                          'n_docs': ndone})
            print(f'  {tag} [curve] step={step_tag} '
                  f'bpb={tot_nll/tot_n:.3f} (docs={ndone})', flush=True)
    if not train_docs:
        print(f'[{tag}] NO train docs — corpus failed entirely', flush=True)
        return {'tag': tag, 'skipped': 'skipped_empty', 'steps_done': 0,
                'curve': [], 'log': []}
    curve_eval(0)
    oom_streak = err_streak = 0
    losses, log, attempt = [], [], -1
    smoke_ok = None if gate else True   # resumed arms are known-good
    t0 = time.time()
    for attempt in range(int(target_ok * 1.75)):
        if len(losses) >= target_ok:
            break
        doc = train_docs[attempt % len(train_docs)]
        try:
            loss = shim_a.train_step(doc)
            if getattr(shim_a, 'nonfinite', False):
                log.append({'attempt': attempt, 'fatal': 'E_nonfinite'})
                print(f'[{tag}] E went NON-FINITE — aborting', flush=True)
                smoke_ok = False
                break
            if loss is None:
                continue
            oom_streak = err_streak = 0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); oom_streak += 1
            log.append({'attempt': attempt, 'oom': True})
            if oom_streak >= 15:
                print(f'[{tag}] 15 consecutive OOMs — aborting', flush=True)
                break
            continue
        except Exception as e:
            err_streak += 1
            log.append({'attempt': attempt, 'error': str(e)[:150]})
            if err_streak >= 8:
                print(f'[{tag}] 8 consecutive errors — aborting', flush=True)
                break
            continue
        losses.append(loss)
        log.append({'attempt': attempt, 'step_ok': len(losses), 'loss': loss})
        n_ok = len(losses)
        if n_ok % 100 == 0:
            print(f'  {tag} ok-step {n_ok} loss={loss:.4f} bits '
                  f'({time.time()-t0:.0f}s)', flush=True)
        if n_ok in curve_at:
            curve_eval(n_ok)
            if gate and n_ok == 50 and smoke_ok is None:
                b0 = next((r['bpb'] for r in curve if r['step'] == 0), None)
                r50 = next((r for r in curve if r['step'] == 50), None)
                b50 = r50['bpb'] if r50 else None
                nan50 = (b50 is None) or (b50 != b50)
                smoke_ok = (b0 is not None and not nan50 and
                            (b0 - b50) >= 0)
                print(f'[{tag}] SMOKE GATE (bpb): step0={b0} '
                      f'step50={b50} -> '
                      f'{"PASS" if smoke_ok else "FAIL"}', flush=True)
                if not smoke_ok:
                    break
        if n_ok % 250 == 0:
            save()
    if not curve or curve[-1]['step'] != len(losses):
        curve_eval('final')
    save()
    print(f'[{tag}] done: {len(losses)}/{target_ok} steps in '
          f'{attempt+1} attempts ({time.time()-t0:.0f}s)', flush=True)
    return {'tag': tag, 'log': log, 'steps_done': len(losses),
            'target': target_ok, 'attempts': attempt + 1,
            'smoke_ok': smoke_ok, 'curve': curve,
            'first_loss': losses[0] if losses else None,
            'last_loss': losses[-1] if losses else None}

# Arm B FIRST — the open question; p1 lr/clip tuned for stability
shim_p1 = AdapterShim(model, tok, period=1, lr=1e-4, clip=0.5)
res_p1 = train_arm(shim_p1, 1500, {50, 250, 750, 1500},
                   '/kaggle/working/adapter_leg4_p1.pt', 'p1')
"""),

    code_cell("""# Cell 7: Arm A RESUME — continue the 3000-step p2 adapter to 6000.
# leg-3 v3 Arm A: 6.964 -> 3.618 bpb, curve still falling at 3000.
# Wall-clock guard: A needs ~3h; skip if >2.5h elapsed (B took ~2h).
if time.time() - T_START > 4.5 * 3600:
    res_p2 = {'tag': 'p2', 'skipped': 'wall_clock_guard', 'steps_done': 0,
              'curve': [], 'log': []}
    print('[p2] SKIPPED — wall-clock guard (>4.5h elapsed)', flush=True)
else:
    CK = '/kaggle/input/twil-leg3-adapter/adapter_leg3_p2.pt'
    try:
        ck = torch.load(CK, map_location='cpu', weights_only=True)
        assert ck['period'] == 2
        shim = AdapterShim(model, tok, period=2)
        shim.E.data.copy_(ck['E'].to(shim.E.device).to(shim.E.dtype))
        print('[p2] resumed from leg-3 3000-step adapter', flush=True)
        res_p2 = train_arm(shim, 3000, {500, 1500, 3000},
                           '/kaggle/working/adapter_leg4_p2.pt', 'p2',
                           gate=False)
    except Exception as e:
        res_p2 = {'tag': 'p2', 'skipped': f'ckpt_fail: {type(e).__name__}',
                  'steps_done': 0, 'curve': [], 'log': []}
        print(f'[p2] checkpoint load FAILED ({type(e).__name__}: '
              f'{str(e)[:150]}) — Arm A skipped', flush=True)
"""),

    code_cell("""# Cell 8: evals — Arm A full (period-2), Arm B curve-subset (period-1),
# per-arm deltas vs own warm-start, effective_bpb sanity (leg-2 gap).
def eval_all(shim_e, docs, mode='learned'):
    out = []
    for dom, doc in docs:
        try:
            bpb, n, (icov, tcov) = shim_e.first_byte_bpb(doc, mode=mode)
            out.append({'domain': dom, 'bpb': bpb, 'n_preds': n,
                        'input_cov': icov, 'target_cov': tcov})
        except Exception as e:
            out.append({'domain': dom,
                        'error': f'{type(e).__name__}: {str(e)[:120]}'})
    ok = [r for r in out if 'error' not in r]
    return out, (round(sum(r['bpb'] for r in ok) / len(ok), 4) if ok
                 else None)

learned_p2, learned_p2_overall = (
    eval_all(shim, eval_docs) if not res_p2.get('skipped')
    else ([], None))
learned_p1, learned_p1_overall = (
    eval_all(shim_p1, eval_docs) if not res_p1.get('skipped')
    else ([], None))
delta_p2 = (round(base_overall - learned_p2_overall, 4)
            if learned_p2_overall is not None else None)
delta_p1 = (round(base_p1_overall - learned_p1_overall, 4)
            if (learned_p1_overall is not None
                and base_p1_overall is not None) else None)
print(f'[leg4] A period-2 resumed full: {learned_p2_overall} '
      f'(delta {delta_p2})', flush=True)
print(f'[leg4] B period-1 full: {learned_p1_overall} '
      f'(delta {delta_p1})', flush=True)

eff = {}
for dom, doc in (eval_docs[:2] if not res_p2.get('skipped') else []):
    for mode in ('v0', 'learned'):
        try:
            pos, cnll, _, _ = shim.boundary_nll(doc, mode)
            bnll = shim.byte_nll(doc, mode)
            eff[f'{dom}/{mode}'] = float(effective_bpb(doc, pos, cnll, bnll))
        except Exception as e:
            eff[f'{dom}/{mode}'] = f'error: {type(e).__name__}: {str(e)[:120]}'
print('[leg4] effective_bpb sanity:', eff, flush=True)

verdict = {'armA_p2': {'learned': learned_p2_overall,
                       'base': base_overall, 'delta': delta_p2,
                       'curve': res_p2.get('curve')},
           'armB_p1': {'learned': learned_p1_overall,
                       'base': base_p1_overall, 'delta': delta_p1,
                       'curve': res_p1.get('curve')},
           'effective_sanity': eff,
           'beats_leg2_delta_0_579': (delta_p2 is not None
                                      and delta_p2 > 0.579)}
print(f'[VERDICT] A delta={delta_p2} B delta={delta_p1} '
      f'beats_leg2={verdict["beats_leg2_delta_0_579"]}', flush=True)
json.dump({'results': learned_p2, 'verdict': verdict},
          open('/kaggle/working/leg4_eval.json', 'w'), indent=1)
json.dump({'results': learned_p1, 'overall': learned_p1_overall},
          open('/kaggle/working/leg4_eval_p1.json', 'w'), indent=1)
json.dump({'p2': res_p2, 'p1': res_p1},
          open('/kaggle/working/leg4_train_log.json', 'w'))
"""),

    code_cell("""# Cell 9: artifacts.zip + run_status.json
import zipfile, os
def arm_status(r):
    if r.get('skipped'):
        return r['skipped']   # 'wall_clock_guard' | 'skipped_empty'
    return 'done' if (r.get('smoke_ok') and
                      r.get('steps_done', 0) >= r.get('target', 0)) \
        else ('partial' if r.get('smoke_ok') else 'aborted')
status = {'kernel': 'twil-leg4', 'lane': 'gpu-2xt4',
          'elapsed_h': round((time.time() - T_START) / 3600, 2),
          'legs': {'load': 'ok', 'baseline': 'done',
                   'armA_p2': arm_status(res_p2),
                   'armB_p1': arm_status(res_p1), 'eval': 'done'},
          'errors': []}
try:
    status['verdict'] = json.load(
        open('/kaggle/working/leg4_eval.json'))['verdict']
except Exception as e:
    status['errors'].append(f'verdict: {e}')
json.dump(status, open('/kaggle/working/run_status.json', 'w'), indent=1)
with zipfile.ZipFile('/kaggle/working/artifacts.zip', 'w') as z:
    for fn in ['corpus_manifest.json', 'leg4_baseline.json',
               'leg4_baseline_p1.json', 'leg4_eval.json',
               'leg4_eval_p1.json', 'leg4_train_log.json',
               'run_status.json', 'adapter_leg4_p2.pt',
               'adapter_leg4_p1.pt']:
        p = f'/kaggle/working/{fn}'
        if os.path.exists(p):
            z.write(p, fn)
print('artifacts.zip written', flush=True)
"""),

]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
json.dump(nb, open(OUT, "w"), indent=1)
print("wrote", OUT)

import ast
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        ast.parse(src)
        print(f"cell {i}: syntax OK ({len(src)} chars)")
