"""Regression tests for glial: composite gate over S1 prob distributions.
Pure functions — no model, no I/O. Run: python -m pytest tests/test_glial.py -q"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.glial import GlialConfig, glial_gate, ROUTE_AUTO, ROUTE_REVIEW, ROUTE_ESCALATE

CFG = GlialConfig()


def test_confident_peaked_auto():
    v = glial_gate({"a": 0.97, "b": 0.02, "c": 0.01})
    assert v.route == ROUTE_AUTO and not v.abstain and not v.forced
    assert v.top == "a" and v.prediction_set == ("a",)
    assert "margin" in v.route_reason


def test_near_tie_escalates():
    v = glial_gate({"a": 0.34, "b": 0.33, "c": 0.33})
    # margin ~0.01 < 0.20 -> escalate (abstain needs conf<0.25 too; here
    # conf=0.34 so it escalates on margin alone)
    assert v.route == ROUTE_ESCALATE


def test_weak_top_abstains():
    # very weak AND near-tied -> abstain suppresses emission
    v = glial_gate({"a": 0.22, "b": 0.21, "c": 0.20, "d": 0.19, "e": 0.18})
    assert v.abstain and v.route == ROUTE_ESCALATE


def test_single_candidate_forced_review():
    # degenerate certainty: conf=1.0 but S1 never discriminated
    v = glial_gate({"only_tool": 1.0})
    assert v.route == ROUTE_REVIEW and v.forced
    assert "forced" in v.route_reason


def test_two_candidates_not_forced():
    v = glial_gate({"a": 0.9, "b": 0.1})
    assert not v.forced and v.route == ROUTE_AUTO


def test_middle_band_review():
    v = glial_gate({"a": 0.60, "b": 0.25, "c": 0.15})  # margin 0.35 in [0.3,0.5)
    assert v.route == ROUTE_REVIEW


def test_empty_probs_escalate_abstain():
    v = glial_gate({})
    assert v.route == ROUTE_ESCALATE and v.abstain and v.top is None


def test_unnormalized_probs_normalized():
    v = glial_gate({"a": 97.0, "b": 2.0, "c": 1.0})
    assert v.route == ROUTE_AUTO and abs(v.conf - 0.97) < 0.01


def test_prediction_set_delta():
    v = glial_gate({"a": 0.80, "b": 0.75, "c": 0.50, "d": 0.40},
                   GlialConfig(set_delta=0.10))
    # a,b within 0.10 of top(0.80->>=0.70); c,d excluded
    assert v.prediction_set == ("a", "b")


def test_route_reason_always_present():
    for probs in ({"a": 1.0}, {"a": 0.5, "b": 0.5}, {},
                  {"a": 0.9, "b": 0.05, "c": 0.05}):
        v = glial_gate(probs)
        assert v.route_reason


def test_config_override():
    strict = GlialConfig(margin_auto=0.95)
    v = glial_gate({"a": 0.80, "b": 0.15, "c": 0.05}, strict)
    # margin 0.65 < 0.95 -> falls to review band (>=0.20)
    assert v.route == ROUTE_REVIEW
