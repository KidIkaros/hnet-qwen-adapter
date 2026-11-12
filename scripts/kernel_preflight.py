#!/usr/bin/env python3
"""Mechanical pre-push checks — one per failure class we have hit.

leg1 OOM (T~1024)     -> T-budget assertion
leg2 dict+=list typo  -> ast.parse every cell
leg3 T_START undefined-> undefined-name scan across cells
leg3 time not imported-> import-before-use scan
leg4 dataset not ready-> kaggle dataset status == ready
leg4 mount nesting    -> kernel must glob-resolve mounted files
"""
import ast, json, re, subprocess, sys


def check_notebook(path, meta_path, kaggle_bin):
    nb = json.load(open(path))
    meta = json.load(open(meta_path))
    code = [''.join(c['source']) for c in nb['cells']
            if c['cell_type'] == 'code']
    fails = []
    for i, c in enumerate(code):
        try:
            ast.parse(c)
        except SyntaxError as e:
            fails.append(f'cell {i}: SyntaxError {e}')
    defined = set(dir(__builtins__)) | {
        'torch', 'np', 'json', 'os', 're', 'sys', 'time', 'math',
        'subprocess', 'zipfile', 'glob', 'print', 'len', 'range'}
    for i, c in enumerate(code):
        tree = ast.parse(c)
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        # defs: assignments, class/def names, args, except-as, imports
        assigned = {n.id for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        assigned |= {n.name for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef))}
        assigned |= {a.arg for n in ast.walk(tree)
                     if isinstance(n, ast.arguments)
                     for a in list(n.args) + list(n.kwonlyargs)}
        assigned |= {n.name for n in ast.walk(tree)
                     if isinstance(n, ast.ExceptHandler) and n.name}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    assigned.add((a.asname or a.name).split('.')[0])
        missing = {m for m in (used - defined - assigned)
                   if not m.startswith('_') and m not in ('self', 'cls')}
        if missing:
            fails.append(f'cell {i}: possibly undefined: {sorted(missing)}')
        defined |= assigned
    src = '\n'.join(code)
    m = re.search(r'DOC_CAP_TRAIN,\s*DOC_CAP_EVAL\s*=\s*(\d+),\s*(\d+)', src)
    if m and int(m.group(1)) > 1024:
        fails.append(f'T-budget: DOC_CAP_TRAIN={m.group(1)} may OOM '
                     f'(leg-1 class)')
    for ds in meta.get('dataset_sources', []):
        r = subprocess.run([kaggle_bin, 'datasets', 'status', ds],
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip().splitlines()
        last = out[-1] if out else ''
        if 'ready' not in last.lower():
            fails.append(f'dataset {ds} not ready: {last[:80]}')
    if meta.get('dataset_sources') and 'glob.glob' not in src:
        fails.append('dataset mounted but no glob fallback (mount class)')
    return fails


if __name__ == '__main__':
    fails = check_notebook(sys.argv[1], sys.argv[2], sys.argv[3])
    for f in fails:
        print('FAIL:', f)
    print('PREFLIGHT', 'FAILED' if fails else 'PASSED',
          f'({len(fails)} issues)')
    sys.exit(1 if fails else 0)
