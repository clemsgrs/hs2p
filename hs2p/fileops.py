from __future__ import annotations

import errno
import shutil
from pathlib import Path


def validate_annotation_name(annotation: str | None) -> str | None:
    """Validate an annotation name, accepting ``None`` for structural output."""
    if annotation is None:
        return None
    if annotation == "merged":
        raise ValueError(
            "annotation name 'merged' is reserved for structural merged coordinate output"
        )
    if (
        not annotation
        or annotation in {".", ".."}
        or any(ch in annotation for ch in ("/", "\\", "\x00"))
    ):
        raise ValueError(
            f"annotation label {annotation!r} must be a safe path component "
            "(non-empty, no '/'\\ separators, not '.' or '..')"
        )
    return annotation


def is_flattened_annotation(annotation: str | None) -> bool:
    """Decide whether an annotation's artifacts land at the flat output root.

    This is the single source of truth for the annotation→path rule shared by the
    coordinate/tar artifact code and the preview/visualization layer: ``None`` and the
    conventional ``"tissue"`` label collapse to the flat layout (no per-annotation subdir),
    while every other label gets its own ``.../{annotation}/...`` location. Structural merged
    output is represented by ``annotation=None`` with ``output_mode="merged"``; the literal
    annotation name ``"merged"`` is reserved and must be rejected at caller boundaries.
    """
    return annotation is None or annotation == "tissue"


def promote_temp_file(temp_path: Path, target_path: Path) -> None:
    """Move a completed temp file into place, with a CIFS-friendly fallback."""
    try:
        temp_path.replace(target_path)
        return
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            raise

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temp_path.open("rb") as source, target_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    except FileNotFoundError:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("rb") as source, target_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    temp_path.unlink(missing_ok=True)
