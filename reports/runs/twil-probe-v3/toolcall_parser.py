import re
CALL_RE = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.S)

def parse_calls(text):
    out = []
    for m in CALL_RE.finditer(text):
        name, body = m.group(1), m.group(2)
        args = {k: v.strip() for k, v in PARAM_RE.findall(body)}
        out.append({"name": name, "args": args})
    return out
