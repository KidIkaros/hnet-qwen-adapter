"""TwIL/Granite tool-call emission parser - the antml scheme.

The format is verified against actual run artifacts (wiring-v1
s2_rows.json, probe-v3 toolcall_results.json): the model emits

    <tool_call>
      <function=name>
        <parameter=key>value
      </parameter>
    </function>
  </tool_call>
</step>

Think traces may contain example markup; callers must parse the
post-think region only (the wiring-v2 lesson: reasoning text
false-positives otherwise).
"""
import re

CALL_RE = re.compile(
    r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>",
    re.S)
PARAM_RE = re.compile(
    r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_calls(text):
    """Emission text -> [{"name": str, "args": {k: v}}, ...]."""
    out = []
    for m in CALL_RE.finditer(text):
        name, body = m.group(1), m.group(2)
        args = {k: v.strip() for k, v in PARAM_RE.findall(body)}
        out.append({"name": name, "args": args})
    return out


def post_think(text):
    """The generation after the think block - the only region where
    calls may be parsed."""
    marker = '</think>'
    return text.rsplit(marker, 1)[-1] if marker in text else text
