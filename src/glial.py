"""glial — composite regulatory gate over S1 decision outputs.

Adopts the laya/vega output contract (route / abstain / prediction_set /
route_reason) but sources every signal from the S1 engine's own
probability distribution — no external model required. Validated on
StartLux-Decision-v1-0.8B ToolACE rows (reports/runs/s1-headtohead-v2 +
reports/runs/startlux-gate-cal-v1/, 220 rows total):

- margin (p1 - p2) separates right from wrong on answerable items with a
  clean gap at 0.5: pooled 169 answerable rows — ALL 15 wrong items have
  margin < 0.5 (max 0.404); 140/154 right items above. Out-of-sample:
  0/7 wrong above 0.5 on the 160-row batch that never set the threshold.
- conf = 1.0 / margin = 1.0 is DEGENERATE when n_candidates == 1: the
  model isn't choosing, it's forced — 7/14 such items had undeclared gold.
  A gate reading only probs would call these "certain"; the contract's
  route_reason exposes them instead.
- abstain never fires on StartLux distributions (conf rarely < 0.25) —
  escalation is carried by the margin bands. Recorded honestly.
- __none__ synthetic-candidate probe: NEGATIVE (AUC 0.58, never chosen) —
  the head ignores it; unbound-gold detection needs a different mechanism.
- Conformal sets collapse to {top} at this accuracy level (94% answerable
  top-1) — coverage == accuracy; the machinery matters when the engine
  is weaker.

Thresholds calibrated on 160 fresh rows (fit/cal) + held-out v2 batch
(eval): margin_auto=0.50, margin_escalate=0.30.
"""
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlialConfig:
    margin_auto: float = 0.50      # margin >= -> route "auto"
    margin_escalate: float = 0.30  # margin <  -> route "escalate"
    abstain_conf: float = 0.25     # conf below AND near-tie -> abstain
    abstain_margin: float = 0.10
    set_delta: float = 0.10        # prediction_set: within delta of top prob
    max_candidates_for_forced: int = 1


ROUTE_AUTO = "auto"
ROUTE_REVIEW = "review"
ROUTE_ESCALATE = "escalate"


@dataclass(frozen=True)
class GlialVerdict:
    route: str
    abstain: bool
    top: str | None
    conf: float
    margin: float
    entropy: float
    n_candidates: int
    prediction_set: tuple = field(default_factory=tuple)
    forced: bool = False
    route_reason: str = ""

    def to_dict(self):
        return {"route": self.route, "abstain": self.abstain,
                "top": self.top, "conf": round(self.conf, 4),
                "margin": round(self.margin, 4),
                "entropy": round(self.entropy, 4),
                "n_candidates": self.n_candidates,
                "prediction_set": list(self.prediction_set),
                "forced": self.forced,
                "route_reason": self.route_reason}


def _features(probs):
    """Sorted (name, p) desc + summary stats. Probs normalized defensively."""
    items = sorted(((str(k), float(v)) for k, v in probs.items()),
                   key=lambda kv: -kv[1])
    total = sum(p for _, p in items)
    if total > 0 and abs(total - 1.0) > 1e-6:
        items = [(n, p / total) for n, p in items]
    ps = [p for _, p in items]
    p1 = ps[0] if ps else 0.0
    p2 = ps[1] if len(ps) > 1 else 0.0
    entropy = -sum(p * math.log(p) for p in ps if p > 0)
    return items, p1, p1 - p2, entropy


def glial_gate(probs, cfg=GlialConfig()):
    """Map an S1 engine's {candidate: prob} distribution to a glial verdict.

 Pure function — the engine boundary owns I/O; this owns the decision
 about what the distribution is worth. Order of checks matters:
 forced-choice and abstain are evaluated before the margin bands because
 a degenerate candidate set makes margin meaningless.
    """
    items, p1, margin, entropy = _features(probs or {})
    n = len(items)
    top = items[0][0] if items else None
    pset = tuple(nm for nm, p in items
                 if p >= p1 - cfg.set_delta) if items else ()

    if n == 0:
        return GlialVerdict(ROUTE_ESCALATE, True, None, 0.0, 0.0, 0.0, 0,
                            (), True, "no candidates — nothing to decide")

    forced = n <= cfg.max_candidates_for_forced
    # abstain: no defensible single pick — near-tie AND weak top prob
    abstain = (p1 < cfg.abstain_conf and margin < cfg.abstain_margin)

    if forced:
        route = ROUTE_REVIEW
        reason = (f"forced choice: {n} candidate(s) — S1 did not "
                  f"discriminate (conf={p1:.3f})")
    elif abstain:
        route = ROUTE_ESCALATE
        reason = (f"no defensible pick: conf={p1:.3f}<{cfg.abstain_conf} "
                  f"and margin={margin:.3f}<{cfg.abstain_margin}")
    elif margin >= cfg.margin_auto:
        route = ROUTE_AUTO
        reason = f"margin={margin:.3f}>={cfg.margin_auto} -> confident"
    elif margin < cfg.margin_escalate:
        route = ROUTE_ESCALATE
        reason = f"margin={margin:.3f}<{cfg.margin_escalate} -> unreliable"
    else:
        route = ROUTE_REVIEW
        reason = (f"margin={margin:.3f} in "
                  f"[{cfg.margin_escalate},{cfg.margin_auto}) -> uncertain")

    return GlialVerdict(route, abstain, top, p1, margin, entropy, n,
                        pset, forced, reason)
