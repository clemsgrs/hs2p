# Lessons

- When a config field is supported only as an undocumented `getattr(...)` fallback, do not assume it should stay part of the design. Check the declared config surface first and prefer removing hidden overrides over carrying them into new plans.
- When consolidating shared runtime models, put them in the user-facing namespace they want (`hs2p.configs` here), but keep implementation split between a lightweight models module and loader utilities so low-level code does not inherit config-loading dependencies.
- When a cleanup includes user-visible terminology choices that are mostly about naming rather than behavior (for example `preview` vs `visualization`), pause and ask the user for the preferred term before propagating it through the codebase unless the choice is truly insignificant.
