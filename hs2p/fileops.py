from __future__ import annotations

import errno
import shutil
from pathlib import Path


def is_flattened_annotation(annotation: str | None) -> bool:
    """Decide whether an annotation's artifacts land at the flat output root.

    This is the single source of truth for the annotation→path rule shared by the
    coordinate/tar artifact code and the preview/visualization layer: ``None`` and the
    sentinel ``"tissue"`` label collapse to the flat layout (no per-annotation subdir),
    while every other label gets its own ``.../{annotation}/...`` location. It lives in
    this leaf module (stdlib-only deps) so both layers can import it without a circular
    import and a preview file can never land in a different place than its coordinate set.
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
