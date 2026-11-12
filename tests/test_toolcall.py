"""Regression tests for the TwIL/Granite tool-call parser.

Fixtures are built with chr(60) so the format is constructed, not typed —
the shape below is the one observed in wiring-v1 s2_rows.json and
probe-v3 toolcall_results.json. The think-trace guard is the wiring-v2
lesson: markup inside <think> must not parse.

Run: python -m pytest tests/test_toolcall.py -q
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.toolcall import parse_calls, post_think

LT = chr(60)


def emit(name, **args):
    body = LT + "tool_call>" + chr(10) + LT + "function=" + name + ">"
    for k, v in args.items():
        body += (chr(10) + LT + "parameter=" + k + ">" + chr(10) + v
                 + chr(10) + LT + "/parameter>")
    body += (chr(10) + LT + "/function>" + chr(10) + LT + "/tool_call>"
             + chr(10) + LT + "/step>")
    return body


def test_single_call_parses():
    calls = parse_calls(emit("get_weather", city="Paris"))
    assert calls == [{"name": "get_weather", "args": {"city": "Paris"}}]


def test_multi_call_parses():
    text = emit("f_one", x="1") + chr(10) + emit("f_two", y="2")
    calls = parse_calls(text)
    assert [c["name"] for c in calls] == ["f_one", "f_two"]
    assert calls[1]["args"] == {"y": "2"}


def test_multiline_value_parses():
    calls = parse_calls(emit("f", code="line1" + chr(10) + "line2"))
    assert calls[0]["args"]["code"] == "line1" + chr(10) + "line2"


def test_no_call_returns_empty():
    assert parse_calls("just prose, no markup") == []


def test_think_trace_markup_does_not_leak():
    # the wiring-v2 failure: reasoning text containing example markup
    text = (LT + "think>" + chr(10) + "maybe " + LT + "function=foo>("
            + LT + "/think>" + chr(10) + emit("bar"))
    calls = parse_calls(post_think(text))
    assert [c["name"] for c in calls] == ["bar"]


def test_post_think_without_think_block_is_identity():
    t = emit("f")
    assert post_think(t) == t
