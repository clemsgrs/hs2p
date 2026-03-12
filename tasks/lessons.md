# Lessons

## 2026-02-14

- For mask contour rendering, do not keep manual thickness overrides when slide-scale consistency is required. Prefer mandatory auto-inference from level-0 WSI dimensions and visualization level.

## 2026-03-12

- When reviewing operational issues, distinguish correctness bugs from observability gaps, but still add lightweight real-time logging for per-item batch failures when the user asks for it so runs are diagnosable before post-hoc CSV inspection.
- When adding temporary-file based atomic writes, always test the failure path too: if the write step raises, explicitly unlink the temporary file so hardening logic does not introduce filesystem leaks.
