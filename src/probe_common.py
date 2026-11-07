"""Shared runtime for kernel/* probe drivers. Replaces the ~45-line
preamble each driver used to duplicate: repo sync, model build, ckpt load,
toolace streaming, forward/score helpers.

Drivers still carry ~10 bootstrap lines (env + git clone + sys.path) since
`colab exec -f` ships only the driver file; this module arrives via the
remote `git pull` of the repo.
"""
import os, sys, glob, subprocess

REPO = "/content/qt/hnet-qwen-adapter"
RESULTS = f"{REPO}/results"
P = 6
CAP = 4096


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:], flush=True)
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def sync_repo():
    if os.path.exists(REPO):
        sh("git", "-C", REPO, "pull", "-q")
    else:
        sh("git", "clone", "-q",
           "https://github.com/KidIkaros/hnet-qwen-adapter.git", REPO)
    sys.path.insert(0, REPO)
    os.chdir(REPO)


def build_model(model=None, enc_layers=6, dec_layers=8, n_fit=4):
    """trunk + H-Net + fitted adapters. Returns (trunk, hn, meta)."""
    import torch                                                    # noqa: F401
    from src.factory import build_hnet, fit_adapters, load_trunk
    trunk = load_trunk(model or os.environ.get("MODEL",
                                               "Qwen/Qwen3.5-2B-Base"))
    hn, meta = build_hnet(trunk, enc_layers=enc_layers,
                          dec_layers=dec_layers)
    docs = [open(f, "rb").read()
            for f in sorted(glob.glob("data/decontam/*.bin"))]
    fit_adapters(hn, trunk, meta, [d[:4096] for d in docs[:n_fit]], P=P)
    return trunk, hn, meta


def load_ckpt(hn, prefix=None):
    """Load the trained adapter weights (glue/enc/dec) into hn."""
    import torch
    prefix = prefix or os.environ.get("CKPT_PREFIX", "hnet_resume_best")
    for d in [RESULTS, "/content/ckpt"]:
        try:
            glue = torch.load(f"{d}/{prefix}_glue.pt", map_location="cuda")
            enc = torch.load(f"{d}/{prefix}_enc.pt", map_location="cuda")
            dec = torch.load(f"{d}/{prefix}_dec.pt", map_location="cuda")
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(f"{prefix}_* not found in {RESULTS} or /content/ckpt")
    hn.w_sup.load_state_dict(glue["w_sup"])
    for i, l in enumerate(hn.enc):
        l.load_state_dict(enc[f"enc_{i}"])
    for i, l in enumerate(hn.dec):
        l.load_state_dict(dec[f"dec_{i}"])
    print(f"[ckpt] loaded {prefix}_* from {d}", flush=True)
    hn.train()
    return d


def forward_bp(hn, trunk, ids, cap=CAP):
    """One forward pass; returns byte logits bp[0]."""
    import torch
    from src.factory import periodic
    x = torch.tensor(ids[-cap:], device="cuda")
    with torch.no_grad():
        bp, _, _ = hn(x, boundaries=periodic(x.shape[0], P), ctx="retrieve",
                      n_ctx=1, super_k=4, head_first=trunk["head"].first)
    return bp


def cand_score(hn, trunk, pid, cand, cap=CAP):
    """Mean logprob of cand bytes after prompt (teacher-forced)."""
    import torch
    ids = (pid + cand)[-cap:]
    with torch.no_grad():
        bp = forward_bp(hn, trunk, ids)
        n0 = len(ids) - len(cand)
        lp = bp[0, n0 - 1:-1].gather(
            1, torch.tensor(ids[n0:], device="cuda").unsqueeze(1)).squeeze(1)
    return float(lp.mean())


def stream_toolace(skip, n_items, want_gold=True, max_tools=24):
    """Yield {sys,user,gold,tools} dicts from ToolACE after doc `skip`."""
    import re
    from datasets import load_dataset
    call_re = re.compile(r"([A-Za-z_][\w]*)\s*\(")
    tool_re = re.compile(r'"name"\s*:\s*"([\w]+)"')
    ds = load_dataset("Team-ACE/ToolACE", split="train", streaming=True)
    out = []
    for i, r in enumerate(ds):
        if i < skip:
            continue
        conv = r.get("conversations") or []
        u = next((c["value"] for c in conv if c.get("from") == "user"), None)
        a = next((c["value"] for c in conv
                  if c.get("from") == "assistant"), None)
        gold = call_re.findall(a) if a else []
        tools = list(dict.fromkeys(tool_re.findall(r["system"])))
        if u and tools and (gold or not want_gold):
            out.append({"sys": r["system"], "user": u,
                        "gold": gold[0] if gold else None,
                        "gold_call": (a or "")[:160],
                        "tools": tools[:max_tools]})
        if len(out) >= n_items:
            break
    return out
