# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — carries the project glossary (trunk, substrate,
  S1, S2, glial, StartLux, one mind) and current experiment state.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in
  (currently `0001-substrate-path.md`, the TwIL substrate decision — note this is
  under active recalibration back to the one-mind vision; flag before relying on it).

If either doesn't exist, **proceed silently** — `/domain-modeling` creates them lazily.

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── docs/adr/
│   └── 0001-substrate-path.md
├── src/        # model code (hnet, adapter, int4, glial)
├── kernel/     # experiment drivers / evals
└── modal/      # Modal launchers
```

## Use the glossary's vocabulary

When your output names a domain concept (issue title, hypothesis, spec, test name),
use the term as defined in `CONTEXT.md` — e.g. "trunk" for the frozen base model,
"S1"/"S2" for the internal depth roles, "glial" for internal regulation — not
synonyms like "backbone", "fast path", or "gate model" that the glossary avoids.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than
silently overriding:

> _Contradicts ADR-0001 (TwIL substrate path) — worth reopening because…_
