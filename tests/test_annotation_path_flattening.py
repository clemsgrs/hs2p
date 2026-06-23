"""The annotation→path flattening rule (None/tissue/merged → flat root, other labels → a
per-annotation subdir) must live in a single shared helper that both the artifact code
and the preview/visualization layer can import without a circular import."""

import pytest

from hs2p.fileops import is_flattened_annotation


@pytest.mark.parametrize("annotation", [None, "tissue", "merged"])
def test_flattened_for_none_tissue_and_merged(annotation):
    # "merged" is the union-of-classes output mode: it carries no single class label, so
    # its artifacts belong at the flat root alongside plain tissue (not a `merged/` subdir).
    assert is_flattened_annotation(annotation) is True


@pytest.mark.parametrize("annotation", ["grade_4", "tumor", "Tissue", "tissue_x", "Merged", "merged_x"])
def test_not_flattened_for_named_labels(annotation):
    assert is_flattened_annotation(annotation) is False


def test_helper_importable_without_circular_import():
    # fileops is a leaf module (stdlib-only deps), so importing the helper from it
    # cannot pull in artifacts or the preview layer.
    import hs2p.fileops as fileops

    assert hasattr(fileops, "is_flattened_annotation")


def test_artifact_dir_uses_shared_helper():
    from hs2p.artifacts import _annotation_tiles_dir

    # When flattened, no per-annotation subdir is appended.
    assert is_flattened_annotation(None)
    assert _annotation_tiles_dir("/out", "tissue") == _annotation_tiles_dir("/out", None)


def test_tar_stem_uses_shared_helper():
    from hs2p.tiling.tar import _annotation_tar_stem

    assert _annotation_tar_stem("s", None) == _annotation_tar_stem("s", "tissue")
    assert _annotation_tar_stem("s", "grade_4") == "s.grade_4.tiles"
