"""Structural output (None) and conventional tissue annotations stay flat, while named
annotations use per-annotation subdirectories."""

import pytest

from hs2p.fileops import is_flattened_annotation


@pytest.mark.parametrize("annotation", [None, "tissue"])
def test_flattened_for_structural_output_and_tissue(annotation):
    # Structural merged output carries annotation=None and output_mode="merged".
    assert is_flattened_annotation(annotation) is True


@pytest.mark.parametrize(
    "annotation",
    ["grade_4", "tumor", "Tissue", "tissue_x", "merged", "Merged", "merged_x"],
)
def test_not_flattened_for_named_labels(annotation):
    assert is_flattened_annotation(annotation) is False


def test_tar_stem_uses_shared_helper():
    from hs2p.tiling.tar import _annotation_tar_stem

    assert _annotation_tar_stem("s", None) == _annotation_tar_stem("s", "tissue")
    assert _annotation_tar_stem("s", "grade_4") == "s.grade_4.tiles"
