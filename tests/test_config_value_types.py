"""Configuration values of the wrong type are rejected instead of coerced.

The typed configs' dataclass annotations are the only type schema. They are enforced when
``TilingConfig``, ``SegmentationConfig``, ``FilterConfig`` and ``PreviewConfig`` are
constructed, so file/CLI loading, the resolvers and programmatic callers share one check.
A string is never read as a bool or a number, a float is never truncated to an int, and a
bool is never read as a number. The only widenings are lossless: an int for a float field,
NumPy scalars as numbers, and a ``str`` or ``os.PathLike`` for a ``Path`` field.
"""

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import hs2p.__main__ as tiling_mod
from hs2p.configs import (
    FilterConfig,
    MaskSamplingConfig,
    PreviewConfig,
    SegmentationConfig,
    TilingConfig,
    default_config,
)
from hs2p.configs.loader import DEFAULT_JPEG_BACKEND
from hs2p.configs.resolvers import (
    resolve_filter_config,
    resolve_mask_sampling_config,
    resolve_preview_config,
    resolve_tiling_config,
)
from hs2p.utils.setup import get_cfg_from_file


def _write_inputs(tmp_path: Path, config_body: str) -> Path:
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text("sample_id,image_path,mask_path\nslide-1,slide-1.svs,\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"csv: {csv_path}\n" + config_body)
    return config_path


@pytest.fixture
def cli(monkeypatch, tmp_path: Path):
    """Run the real CLI entrypoint (argument parsing, file + override merge, setup,
    resolvers) and fail loudly if a rejected config ever reaches tiling."""

    def _tile_slides_must_not_run(*args, **kwargs):
        raise AssertionError("tile extraction started for a rejected config")

    monkeypatch.setattr(tiling_mod, "tile_slides", _tile_slides_must_not_run)
    output_dir = tmp_path / "output"

    def run(config_path: Path, *overrides: str):
        return tiling_mod.entrypoint(
            [
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--skip-datetime",
                "--skip-logging",
                *overrides,
            ]
        )

    run.output_dir = output_dir
    return run


def test_cli_rejects_an_unknown_masks_strategy(cli, tmp_path):
    config_path = _write_inputs(tmp_path, 'tiling:\n  masks:\n    strategy: "no"\n')

    with pytest.raises(ValueError) as excinfo:
        cli(config_path)

    assert (
        "Config value tiling.masks.strategy must be one of 'joint', 'independent', "
        "got str 'no'"
    ) in str(excinfo.value)


@pytest.mark.parametrize(
    ("config_body", "overrides", "expected"),
    [
        pytest.param(
            'tiling:\n  params:\n    requested_spacing_um: "0.5"\n',
            (),
            "Config value tiling.params.requested_spacing_um must be a float, "
            "got str '0.5'",
            id="string-for-float",
        ),
        pytest.param(
            "",
            ("tiling.params.requested_tile_size_px=256.7",),
            "Config value tiling.params.requested_tile_size_px must be an int, "
            "got float 256.7",
            id="float-for-int-override",
        ),
        pytest.param(
            'tiling:\n  preview:\n    save_mask_preview: "no"\n',
            (),
            "Config value tiling.preview.save_mask_preview must be a bool, got str 'no'",
            id="string-for-preview-bool",
        ),
        pytest.param(
            'tiling:\n  filter_params:\n    filter_white: "false"\n',
            (),
            "Config value tiling.filter_params.filter_white must be a bool, "
            "got str 'false'",
            id="string-for-filter-bool",
        ),
        pytest.param(
            "",
            ("tiling.filter_params.a_t=abc",),
            "Config value tiling.filter_params.a_t must be an int, got str 'abc'",
            id="string-for-filter-int-override",
        ),
        pytest.param(
            'tiling:\n  seg_params:\n    sthresh: "8"\n',
            (),
            "Config value tiling.seg_params.sthresh must be an int, got str '8'",
            id="string-for-segmentation-int",
        ),
    ],
)
def test_cli_rejects_wrong_typed_values_before_any_slide_is_tiled(
    cli, tmp_path, config_body, overrides, expected
):
    config_path = _write_inputs(tmp_path, config_body)

    with pytest.raises(ValueError) as excinfo:
        cli(config_path, *overrides)

    assert expected in str(excinfo.value)


def test_one_error_lists_every_wrong_typed_field_of_a_section(cli, tmp_path):
    config_path = _write_inputs(
        tmp_path,
        "tiling:\n"
        "  params:\n"
        '    requested_spacing_um: "0.5"\n'
        "    tolerance: true\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cli(config_path, "tiling.params.requested_tile_size_px=256.7")

    assert str(excinfo.value).splitlines() == [
        "Config value tiling.params.requested_spacing_um must be a float, got str '0.5'",
        "Config value tiling.params.requested_tile_size_px must be an int, "
        "got float 256.7",
        "Config value tiling.params.tolerance must be a float, got bool True",
    ]


# Direct construction: programmatic callers (slide2vec, soma) get the same check, with the
# dataclass field named in the error.


def _tiling(**overrides) -> TilingConfig:
    values = dict(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.05,
        overlap=0.0,
    )
    return TilingConfig(**{**values, **overrides})


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        pytest.param(
            lambda: PreviewConfig(save_mask_preview="no"),
            "Config value PreviewConfig.save_mask_preview must be a bool, got str 'no'",
            id="str-for-bool",
        ),
        pytest.param(
            lambda: FilterConfig(filter_white=1),
            "Config value FilterConfig.filter_white must be a bool, got int 1",
            id="int-for-bool",
        ),
        pytest.param(
            lambda: FilterConfig(a_t=4.0),
            "Config value FilterConfig.a_t must be an int, got float 4.0",
            id="integral-float-for-int",
        ),
        pytest.param(
            lambda: FilterConfig(white_threshold=True),
            "Config value FilterConfig.white_threshold must be an int, got bool True",
            id="bool-for-int",
        ),
        pytest.param(
            lambda: FilterConfig(blur_threshold=np.bool_(False)),
            "Config value FilterConfig.blur_threshold must be a float, "
            f"got {type(np.bool_(False)).__name__} {np.bool_(False)!r}",
            id="numpy-bool-for-float",
        ),
        pytest.param(
            lambda: _tiling(overlap="0"),
            "Config value TilingConfig.overlap must be a float, got str '0'",
            id="str-for-float",
        ),
        pytest.param(
            lambda: SegmentationConfig(method=None),
            "Config value SegmentationConfig.method must be a str, got NoneType None",
            id="none-for-str",
        ),
        pytest.param(
            lambda: SegmentationConfig(method="hsv", sam2_num_workers="2"),
            "Config value SegmentationConfig.sam2_num_workers must be an int or None, "
            "got str '2'",
            id="str-for-optional-int",
        ),
        pytest.param(
            lambda: SegmentationConfig(method="sam2", sam2_checkpoint_path=3),
            "Config value SegmentationConfig.sam2_checkpoint_path must be a path or None, "
            "got int 3",
            id="int-for-optional-path",
        ),
        pytest.param(
            lambda: PreviewConfig(tissue_contour_color=(37, 94)),
            "Config value PreviewConfig.tissue_contour_color must be a tuple of 3 ints, "
            "got tuple (37, 94)",
            id="short-tuple",
        ),
        pytest.param(
            lambda: PreviewConfig(tissue_contour_color=(37.0, 94, 59)),
            "Config value PreviewConfig.tissue_contour_color must be a tuple of 3 ints, "
            "got tuple (37.0, 94, 59)",
            id="float-in-int-tuple",
        ),
        pytest.param(
            lambda: PreviewConfig(tissue_contour_color="red"),
            "Config value PreviewConfig.tissue_contour_color must be a tuple of 3 ints, "
            "got str 'red'",
            id="str-for-tuple",
        ),
        pytest.param(
            lambda: MaskSamplingConfig(pixel_mapping={"tumor": 1}, min_coverage=0.1),
            "Config value MaskSamplingConfig.min_coverage must be a mapping, got float 0.1",
            id="float-for-mapping",
        ),
        pytest.param(
            lambda: MaskSamplingConfig(
                pixel_mapping={"tumor": 1}, min_coverage={}, strategy="independent_sampling"
            ),
            "Config value MaskSamplingConfig.strategy must be one of 'joint', "
            "'independent', got str 'independent_sampling'",
            id="unknown-literal",
        ),
    ],
)
def test_direct_construction_rejects_wrong_typed_values(build, expected):
    with pytest.raises(ValueError) as excinfo:
        build()

    assert str(excinfo.value) == expected


def test_direct_construction_widens_numbers_losslessly():
    tiling = _tiling(
        requested_spacing_um=np.float32(0.5),
        requested_tile_size_px=np.int64(224),
        tolerance=np.float64(0.05),
        overlap=0,
    )

    assert tiling.requested_spacing_um == 0.5
    assert type(tiling.requested_spacing_um) is float
    assert tiling.requested_tile_size_px == 224
    assert type(tiling.requested_tile_size_px) is int
    assert type(tiling.tolerance) is float
    assert tiling.overlap == 0.0
    assert type(tiling.overlap) is float


class _PathLike(os.PathLike):
    def __init__(self, path: str) -> None:
        self._path = path

    def __fspath__(self) -> str:
        return self._path


def test_direct_construction_accepts_strings_and_pathlikes_for_path_fields():
    segmentation = SegmentationConfig(
        method="sam2",
        sam2_checkpoint_path="weights/sam2.pt",
        sam2_config_path=_PathLike("configs/sam2.yaml"),
        sam2_num_workers=np.int16(1),
    )

    assert segmentation.sam2_checkpoint_path == Path("weights/sam2.pt")
    assert segmentation.sam2_config_path == Path("configs/sam2.yaml")
    assert segmentation.sam2_num_workers == 1


def test_direct_construction_stores_a_list_colour_as_a_tuple():
    preview = PreviewConfig(tissue_contour_color=[np.uint8(1), 2, 3])

    assert preview.tissue_contour_color == (1, 2, 3)
    assert type(preview.tissue_contour_color) is tuple


# An interpolation that fills a scalar field with a mapping is a wrong-typed value too.


def test_cli_rejects_an_interpolation_that_fills_a_bool_with_a_mapping(cli, tmp_path):
    """An empty mapping has no keys for the unknown-key check to report, and used to be
    read as ``False``."""
    config_path = _write_inputs(
        tmp_path,
        "tiling:\n  filter_params:\n    filter_white: ${wandb.flags}\n"
        "wandb:\n  flags: {}\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cli(config_path)

    assert (
        "Config value tiling.filter_params.filter_white must be a bool, got DictConfig {}"
        in str(excinfo.value)
    )


@pytest.mark.parametrize(
    ("resolve", "path"),
    [
        (resolve_filter_config, "tiling.filter_params.filter_white"),
        (resolve_preview_config, "tiling.preview.save_mask_preview"),
    ],
    ids=["resolve_filter_config", "resolve_preview_config"],
)
def test_resolvers_reject_an_interpolation_that_fills_a_bool_with_a_mapping(
    resolve, path
):
    """A hand-built, unresolved DictConfig passed straight to a resolver: the interpolated
    mapping reaches the check instead of being read as ``True``."""
    cfg = OmegaConf.create(
        {
            "tiling": OmegaConf.to_container(default_config.tiling, resolve=False),
            "wandb": {"flags": {"enabled": False}},
        }
    )
    OmegaConf.update(cfg, path, "${wandb.flags}")

    with pytest.raises(ValueError) as excinfo:
        resolve(cfg)

    assert (
        f"Config value {path} must be a bool, got DictConfig {{'enabled': False}}"
        in str(excinfo.value)
    )


def test_mask_sampling_resolver_rejects_a_bool_for_the_strategy():
    cfg = SimpleNamespace(
        tiling=SimpleNamespace(
            masks=SimpleNamespace(
                pixel_mapping={"tumor": 1}, min_coverage={"tumor": 0.5}, strategy=True
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^Config value tiling\.masks\.strategy must be one of 'joint', "
            r"'independent', got bool True$"
        ),
    ):
        resolve_mask_sampling_config(cfg)


# Top-level and ``speed`` scalars that the CLI reads outside the typed configs are checked at
# loading, against types declared the same way.


def test_cli_rejects_wrong_typed_run_settings_at_loading(cli, tmp_path):
    config_path = _write_inputs(
        tmp_path,
        'resume: "no"\nsave_tiles: "false"\nseed: "0"\n',
    )

    with pytest.raises(ValueError) as excinfo:
        cli(config_path, "speed.num_workers=2.5", "speed.jpeg_backend=1")

    assert str(excinfo.value).splitlines() == [
        "Config value resume must be a bool, got str 'no'",
        "Config value save_tiles must be a bool, got str 'false'",
        "Config value seed must be an int, got str '0'",
        "Config value speed.num_workers must be an int, got float 2.5",
        "Config value speed.jpeg_backend must be a str, got int 1",
    ]
    # Rejected at loading: setup never created the run's output directory.
    assert not cli.output_dir.exists()


def test_file_loading_rejects_wrong_typed_run_settings(tmp_path):
    config_path = _write_inputs(
        tmp_path, "speed:\n  num_workers: ${wandb.workers}\nwandb:\n  workers: four\n"
    )

    with pytest.raises(
        ValueError,
        match=r"^Config value speed\.num_workers must be an int, got str 'four'$",
    ):
        get_cfg_from_file(config_path)


def test_cli_keeps_the_jpeg_default_when_an_interpolated_speed_section_omits_it(
    monkeypatch, tmp_path
):
    """An interpolated ``speed`` mapping replaces the default section, so its absent
    ``jpeg_backend`` is not a wrong-typed value: the CLI falls back to the default."""
    received = {}

    def _record(whole_slides, **kwargs):
        received.update(kwargs)
        (kwargs["output_dir"] / "process_list.csv").write_text("")
        return []

    monkeypatch.setattr(tiling_mod, "tile_slides", _record)
    config_path = _write_inputs(
        tmp_path, "speed: ${wandb.speed}\nwandb:\n  speed:\n    num_workers: 2\n"
    )

    tiling_mod.entrypoint(
        [
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-datetime",
            "--skip-logging",
        ]
    )

    assert received["num_workers"] == 2
    assert received["jpeg_backend"] == DEFAULT_JPEG_BACKEND


def test_file_loading_rejects_an_explicit_null_run_setting(tmp_path):
    config_path = _write_inputs(tmp_path, "speed:\n  jpeg_backend: null\n")

    with pytest.raises(
        ValueError,
        match=r"^Config value speed\.jpeg_backend must be a str, got NoneType None$",
    ):
        get_cfg_from_file(config_path)


def test_wandb_values_are_not_type_checked(tmp_path):
    config_path = _write_inputs(tmp_path, "wandb:\n  project: 3\n  tags: x\n")

    cfg = get_cfg_from_file(config_path)

    assert cfg.wandb.project == 3
    assert cfg.wandb.tags == "x"


# Values that are valid keep working, from files and from programmatic callers.


def test_cli_accepts_integers_for_float_fields(monkeypatch, tmp_path):
    """YAML ``overlap: 0`` is an int; it widens to the float field losslessly."""
    received = {}

    def _record(whole_slides, **kwargs):
        received.update(kwargs)
        (kwargs["output_dir"] / "process_list.csv").write_text("")
        return []

    monkeypatch.setattr(tiling_mod, "tile_slides", _record)
    config_path = _write_inputs(
        tmp_path, "tiling:\n  params:\n    requested_spacing_um: 1\n    overlap: 0\n"
    )

    tiling_mod.entrypoint(
        [
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-datetime",
            "--skip-logging",
            "tiling.filter_params.blur_threshold=50",
        ]
    )

    tiling = received["tiling"]
    assert tiling.requested_spacing_um == 1.0
    assert type(tiling.requested_spacing_um) is float
    assert type(tiling.overlap) is float
    assert type(received["filtering"].blur_threshold) is float


def test_resolvers_accept_numpy_scalars_and_integers_from_programmatic_adapters():
    """Shaped like slide2vec's adapter, with values computed by NumPy."""
    cfg = SimpleNamespace(
        tiling=SimpleNamespace(
            params=SimpleNamespace(
                requested_spacing_um=np.float64(0.5),
                requested_tile_size_px=np.int64(224),
                tolerance=0,
                overlap=np.float32(0.25),
            ),
            backend="auto",
            mask_backend="auto",
            preview=SimpleNamespace(
                save_mask_preview=True,
                save_tiling_preview=np.bool_(False),
                downsample=np.int32(32),
                tissue_contour_color=[37, 94, 59],
                mask_overlay_alpha=1,
            ),
        )
    )

    tiling = resolve_tiling_config(cfg)
    preview = resolve_preview_config(cfg)

    assert (tiling.requested_spacing_um, tiling.requested_tile_size_px) == (0.5, 224)
    assert (tiling.tolerance, tiling.overlap) == (0.0, 0.25)
    assert preview == PreviewConfig(
        save_mask_preview=True,
        save_tiling_preview=False,
        downsample=32,
        tissue_contour_color=(37, 94, 59),
        mask_overlay_alpha=1.0,
    )


# The annotations drive the check: every field is covered without per-field code.

_REQUIRED = {
    TilingConfig: dict(
        requested_spacing_um=0.5,
        requested_tile_size_px=224,
        tolerance=0.05,
        overlap=0.0,
    ),
    MaskSamplingConfig: dict(pixel_mapping={"tumor": 1}, min_coverage={}),
    SegmentationConfig: dict(method="hsv"),
    FilterConfig: {},
    PreviewConfig: {},
}


@pytest.mark.parametrize(
    ("config_cls", "field_name"),
    [
        (config_cls, field.name)
        for config_cls in _REQUIRED
        for field in dataclasses.fields(config_cls)
    ],
)
def test_every_typed_config_field_is_checked(config_cls, field_name):
    with pytest.raises(ValueError) as excinfo:
        config_cls(**{**_REQUIRED[config_cls], field_name: object()})

    assert str(excinfo.value).startswith(
        f"Config value {config_cls.__name__}.{field_name} must be "
    )
