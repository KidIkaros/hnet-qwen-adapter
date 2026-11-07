"""Jev decision layer — OFFLINE, two-signal fusion. Loads jev_scores.json
(byte-model logprob scores) + enc_scores.json (encoder cosine sims), fuses
via per-signal temperature + convex weight fit on the calibrate split, then
emits nanocore-style verdicts (Choice / Clarify / Noul) with conformal sets.

    python3 kernel/jev_decide.py [jev.json] [enc.json] [alpha] [precision]
"""
import sys, json
import numpy as np

NANOCORE = "/home/ikaaros/Coding/Gold/nanocore-s1"
sys.path.insert(0, NANOCORE)
from src.decision import protocol, scoring  # noqa: E402

JEV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/jev_scores.json"
ENC = sys.argv[2] if len(sys.argv) > 2 else "/tmp/enc_scores.json"
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
TARGET = float(sys.argv[4]) if len(sys.argv) > 4 else 0.85
KMAX = 24


def pad_score_rows(rows, key):
    n = len(rows)
    S = np.full((n, KMAX), -np.inf)
    for i, r in enumerate(rows):
        S[i, :len(r[key])] = r[key]
    return S


def gold_indices(rows):
    out = np.full(len(rows), -1, dtype=int)
    for i, r in enumerate(rows):
        if r["gold"] in r["tools"]:
            out[i] = r["tools"].index(r["gold"])
    return out


jev_rows, enc_rows = json.load(open(JEV)), json.load(open(ENC))
assert all(a["tools"] == b["tools"]
           for a, b in zip(jev_rows, enc_rows)), "tool lists differ"
S_byte, S_enc = pad_score_rows(jev_rows, "scores"), pad_score_rows(enc_rows, "sims")
gold = gold_indices(jev_rows)
keep = gold >= 0
dropped = int((~keep).sum())
if dropped:
    print(f"[data] dropped {dropped} items with undeclared gold", flush=True)
S_byte, S_enc, gold = S_byte[keep], S_enc[keep], gold[keep]
n = len(gold)

rng = np.random.default_rng(7)
perm = rng.permutation(n)
th = n // 3
FIT, CAL, EV = perm[:th], perm[th:2 * th], perm[2 * th:]


def onehot(idx):
    t = np.zeros((len(idx), KMAX)); t[np.arange(len(idx)), idx] = 1.0
    return t


# per-signal temperature (fit split)
T_b = protocol.fit_temperature(S_byte[FIT], onehot(gold[FIT]))
T_e = protocol.fit_temperature(S_enc[FIT], onehot(gold[FIT]))
print(f"[calib] T_byte={T_b:.4f} T_enc={T_e:.4f}", flush=True)


def probs(S, T, idx):
    return protocol.softmax(S[idx], T)


# fusion weight: convex combo of calibrated probs, fit on CAL by log-loss
P_b_cal = probs(S_byte, T_b, CAL)
P_e_cal = probs(S_enc, T_e, CAL)
y_cal = gold[CAL]
best_w, best_ll = 0.0, np.inf
for w in np.linspace(0, 1, 41):
    P = w * P_b_cal + (1 - w) * P_e_cal
    ll = -np.log(np.clip(P[np.arange(len(y_cal)), y_cal], 1e-12, 1)).mean()
    if ll < best_ll:
        best_ll, best_w = ll, w
print(f"[fusion] w_byte={best_w:.2f} (cal logloss {best_ll:.3f})", flush=True)

# eval: per-signal + fused
P_b = probs(S_byte, T_b, EV)
P_e = probs(S_enc, T_e, EV)
P_f = best_w * P_b + (1 - best_w) * P_e
y_ev = gold[EV]
for name, P in [("byte", P_b), ("enc", P_e), ("fused", P_f)]:
    acc = (P.argmax(1) == y_ev).mean()
    print(f"[eval] {name:5s} argmax acc {acc:.3f} "
          f"({int((P.argmax(1)==y_ev).sum())}/{len(EV)})", flush=True)

# conformal + verdicts on fused
S_f = np.log(np.clip(P_f, 1e-12, 1))          # back to score space for aps
S_f_cal = np.log(np.clip(best_w * P_b_cal + (1 - best_w) * P_e_cal,
                         1e-12, 1))
t_set = scoring.fit_set_temperature(S_f_cal, y_cal, ALPHA)
qhat = scoring.qhat_from_scores(S_f_cal, y_cal, t_set, ALPHA)
members = scoring.aps_members(scoring.softmax_rows(S_f, t_set), qhat)
covered = np.mean([y_ev[i] in members[i] for i in range(len(EV))])
sizes = np.array([len(m) for m in members])
print(f"[conformal] coverage {covered:.3f} (target {1-ALPHA:.2f}) "
      f"| mean set {sizes.mean():.1f}", flush=True)

conf = P_f.max(1); correct = P_f.argmax(1) == y_ev

# ---- routed tier: S1 (enc) answers when top-2 sim margin is decisive,
# else escalate to S2 (byte scorer). tau fitted on CAL.
def margins(S, idx):
    out = np.zeros(len(idx))
    for j, i in enumerate(idx):
        k = int(np.isfinite(S[i]).sum())
        s = np.sort(S[i, :k])[::-1]
        out[j] = (s[0] - s[1]) if k > 1 else 9.9
    return out


def routed_acc(tau, m, P_e, P_b, y):
    picks = np.where(m >= tau,
                     P_e.argmax(1), P_b.argmax(1))
    return (picks == y).mean(), float((m < tau).mean())


m_cal = margins(S_enc, CAL)
m_ev = margins(S_enc, EV)
P_b_cal = probs(S_byte, T_b, CAL)

best_tau, best_acc, best_esc = 0.0, -1, 0
for tau in np.linspace(0, 0.08, 33):
    a, e = routed_acc(tau, m_cal, P_e_cal, P_b_cal, y_cal)
    if a > best_acc or (a == best_acc and e < best_esc):
        best_tau, best_acc, best_esc = tau, a, e
P_b_ev = probs(S_byte, T_b, EV)
acc_r, esc_r = routed_acc(best_tau, m_ev, P_e, P_b_ev, y_ev)
print(f"[route] tau={best_tau:.3f} | routed acc {acc_r:.3f} "
      f"(esc rate {esc_r:.2f}) vs enc {float((P_e.argmax(1)==y_ev).mean()):.3f}",
      flush=True)

thr = protocol.coverage_at_precision(P_f, onehot(y_ev), TARGET)
print(f"[verdicts] flat fused acc {correct.mean():.3f} | curve:", flush=True)
for c in protocol.selective_curve(P_f, onehot(y_ev)):
    print(f"    cov {c['coverage']:.2f} -> sel_acc "
          f"{c['selective_accuracy']:.3f}", flush=True)
print(f"  coverage@{TARGET}: {thr['coverage']} (saturated={thr['saturated']})",
      flush=True)
