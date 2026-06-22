"""The annotation label → tiles directory boundary must reject path-traversal labels for
every caller (the public tile_slides(sampling=...) API bypasses config-level validation)."""

from pathlib import Path

import pytest

from hs2p.artifacts import _annotation_tiles_dir


def test_annotation_tiles_dir_flat_layout_for_none_and_tissue(tmp_path):
    base = tmp_path / "tiles"
    assert _annotation_tiles_dir(tmp_path, None) == base
    assert _annotation_tiles_dir(tmp_path, "tissue") == base


def test_annotation_tiles_dir_subdir_for_named_class(tmp_path):
    assert _annotation_tiles_dir(tmp_path, "grade_4") == tmp_path / "tiles" / "grade_4"


@pytest.mark.parametrize("bad", ["../outside", "/tmp/owned", "a/b", "..", ".", "x\\y", ""])
def test_annotation_tiles_dir_rejects_traversal(tmp_path, bad):
    with pytest.raises(ValueError, match="path component"):
        _annotation_tiles_dir(tmp_path, bad)
