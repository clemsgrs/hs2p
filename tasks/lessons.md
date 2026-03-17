# Lessons

- When a config field is supported only as an undocumented `getattr(...)` fallback, do not assume it should stay part of the design. Check the declared config surface first and prefer removing hidden overrides over carrying them into new plans.
