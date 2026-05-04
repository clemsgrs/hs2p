from __future__ import annotations

import errno
import shutil
from pathlib import Path


def promote_temp_file(temp_path: Path, target_path: Path) -> None:
    """Move a completed temp file into place, with a CIFS-friendly fallback."""
    try:
        temp_path.replace(target_path)
        return
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EPERM}:
            raise

    with temp_path.open("rb") as source, target_path.open("wb") as target:
        shutil.copyfileobj(source, target)
