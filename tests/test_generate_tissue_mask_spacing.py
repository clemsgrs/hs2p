import pytest

from scripts.generate_tissue_mask import _resolve_spacing_plan


class _SpacingPlanWSI:
    def __init__(self, spacings: list[float], shapes: list[tuple[int, int]]) -> None:
        self.spacings = spacings
        self.shapes = shapes


def test_resolve_spacing_plan_resamples_directly_when_target_exceeds_coarsest_native_level():
    wsi = _SpacingPlanWSI(
        spacings=[0.24, 0.48, 0.96, 1.92],
        shapes=[(1024, 1024), (512, 512), (256, 256), (128, 128)],
    )

    level, read_spacing, effective_spacing, needs_resampling = _resolve_spacing_plan(
        wsi=wsi,
        target_spacing=4.0,
        tolerance=0.05,
        spacing_at_level_0=None,
    )

    assert level == 3
    assert read_spacing == pytest.approx(1.92)
    assert effective_spacing == pytest.approx(4.0)
    assert needs_resampling is True


def test_resolve_spacing_plan_error_lists_target_available_spacings_and_guidance():
    wsi = _SpacingPlanWSI(
        spacings=[0.24, 0.48, 0.96, 1.92],
        shapes=[(1024, 1024), (512, 512), (256, 256), (128, 128)],
    )

    with pytest.raises(ValueError) as exc_info:
        _resolve_spacing_plan(
            wsi=wsi,
            target_spacing=0.20,
            tolerance=0.05,
            spacing_at_level_0=None,
        )

    message = str(exc_info.value).lower()
    assert "target spacing (0.2" in message
    assert "available spacings" in message
    assert "0.24" in message
    assert "increasing the tolerance" in message
