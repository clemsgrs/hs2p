# Lessons

## 2026-02-14 - Config Container Compatibility

- Mistake pattern: validating config values with concrete container types (`list`/`tuple`) rejected OmegaConf containers (e.g., `ListConfig`) used in real runs.
- Prevention rule: when validating values sourced from config, validate by interface/shape (sequence length + element type/range), not by specific concrete container class.
- Testing rule: add at least one test with OmegaConf container types for any new config validator.
