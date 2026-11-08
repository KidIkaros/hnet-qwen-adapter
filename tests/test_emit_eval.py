"""Regression tests for emit_eval: parse_call, tool_schemas, score_emission.
Pure functions — no model, no I/O. Run: python -m pytest tests/ -q"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.emit_eval import (dup_class, parse_call, tool_schemas,
                           score_emission)

SYS = ('{"name": "get_weather", "description": "w", "parameters": '
       '{"type": "object", "properties": {"city": {"type": "string"}, '
       '"units": {"type": "string", "enum": ["c", "f"]}}, '
       '"required": ["city"]}}, '
       '{"name": "send_msg", "description": "m", "parameters": '
       '{"properties": {"to": {"type": "string"}, "body": {"type": "string"}}}}')
TOOLS = ["get_weather", "send_msg"]


def test_parse_simple():
    c = parse_call(b'get_weather(city="Paris", units="c")]')
    assert c["name"] == "get_weather" and c["closed"]
    assert c["args"] == [("city", '"Paris"'), ("units", '"c"')]


def test_parse_nested_and_quotes():
    c = parse_call('f(a="x(y", b=func(1, 2))')
    assert c["name"] == "f" and c["closed"]
    assert c["args"][0] == ("a", '"x(y"')


def test_parse_truncated():
    c = parse_call(b'get_weather(city="Par')
    assert c["name"] == "get_weather" and not c["closed"]
    assert c["args"] == [("city", '"Par')]


def test_parse_none():
    assert parse_call(b"no call here") is None
    assert parse_call(b"") is None


def test_dup_class_unchanged():
    assert dup_class(b'a="1",a="2"')[1] == "argdup"
    assert dup_class(b"x" * 300)[1] == "runaway"
    assert dup_class(b'clean(arg="v")]')[0] is None


def test_tool_schemas():
    s = tool_schemas(SYS)
    assert s["get_weather"] == frozenset({"city", "units"})
    assert s["send_msg"] == frozenset({"to", "body"})
    assert tool_schemas('{"name": "bare", "description": "d"}')["bare"] is None


def test_score_good_call():
    v = score_emission(b'get_weather(city="Paris", units="c")]',
                       tools=TOOLS, sys_text=SYS,
                       gold_name="get_weather",
                       gold_call='get_weather(city="Paris", units="c")')
    assert v["parse_ok"] and v["tool_valid"] and v["args_valid"]
    assert v["gold_match"] and v["closed"] and v["phenotype"] is None


def test_score_wrong_tool_and_args():
    v = score_emission(b'nope(city="X", bogus="1")]', tools=TOOLS,
                       sys_text=SYS, gold_name="get_weather")
    assert v["tool_valid"] is False and v["gold_match"] is False
    assert v["args_valid"] is None                    # tool not in schemas
    v2 = score_emission(b'get_weather(city="X", wrong="1")]',
                        tools=TOOLS, sys_text=SYS, gold_name="get_weather")
    assert v2["tool_valid"] and v2["args_valid"] is False


def test_score_unparseable_and_loop():
    v = score_emission(b'garbage no call' + b'x' * 300,
                       tools=TOOLS, sys_text=SYS, gold_name="g")
    assert v["parse_ok"] is False and v["tool_valid"] is None
    assert v["gold_match"] is False and v["phenotype"] == "runaway"


def test_gold_fallback_name_match():
    v = score_emission(b'send_msg(to="a@b", body="hi")]', tools=TOOLS,
                       sys_text=SYS, gold_name="send_msg")
    assert v["gold_match"] is True
