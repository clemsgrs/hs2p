"""Unknown configuration keys are rejected instead of silently ignored.

``default.yaml`` is the only schema: every key path present in a config must exist there,
except under the three open ``tiling.masks`` label maps and under ``wandb``. Missing keys are
never an error. The check runs at file/CLI loading and in the resolvers that programmatic
callers use.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import hs2p.__main__ as tiling_mod
from hs2p.configs import default_config
from hs2p.configs.resolvers import (
    resolve_filter_config,
    resolve_preview_config,
    resolve_sampling_request,
    resolve_sampling_spec,
    resolve_segmentation_config,
    resolve_tiling_config,
)
from hs2p.utils.setup import get_cfg_from_args, get_cfg_from_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_inputs(tmp_path: Path, config_body: str) -> Path:
    csv_path = tmp_path / "slides.csv"
    csv_path.write_text("sample_id,image_path,mask_path\nslide-1,slide-1.svs,\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"csv: {csv_path}\n" + config_body)
    return config_path


@pytest.fixture
def cli(monkeypatch, tmp_path: Path):
    """Run the real CLI entrypoint (argument parsing, file + override merge, setup) and fail
    loudly if a rejected config ever reaches tiling."""

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


def test_cli_rejects_misspelled_file_key_before_any_slide_is_tiled(cli, tmp_path):
    config_path = _write_inputs(tmp_path, "tiling:\n  params:\n    spacng: 0.5\n")

    with pytest.raises(ValueError) as excinfo:
        cli(config_path)

    assert (
        "Unknown config key tiling.params.spacng "
        "(did you mean tiling.params.requested_spacing_um?)"
    ) in str(excinfo.value)
    # Rejected at loading: setup never created the run's output directory.
    assert not cli.output_dir.exists()


def test_cli_rejects_misspelled_override_before_any_slide_is_tiled(cli, tmp_path):
    config_path = _write_inputs(tmp_path, "")

    with pytest.raises(ValueError) as excinfo:
        cli(config_path, "speed.num_worker=4")

    assert (
        "Unknown config key speed.num_worker (did you mean speed.num_workers?)"
        in str(excinfo.value)
    )
    assert not cli.output_dir.exists()


def test_one_error_reports_every_unknown_key(cli, tmp_path):
    config_path = _write_inputs(
        tmp_path,
        "tiling:\n"
        "  params:\n"
        "    spacng: 0.5\n"
        "  masks:\n"
        "    min_coverag:\n"
        "      tissue: 0.1\n"
        "  seg_params:\n"
        "    sthres: 10\n"
        "frobnicate: true\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cli(config_path, "speed.num_worker=4")

    message = str(excinfo.value)
    for expected in (
        "Unknown config key tiling.params.spacng "
        "(did you mean tiling.params.requested_spacing_um?)",
        "Unknown config key tiling.masks.min_coverag "
        "(did you mean tiling.masks.min_coverage?)",
        "Unknown config key tiling.seg_params.sthres "
        "(did you mean tiling.seg_params.sthresh?)",
        "Unknown config key speed.num_worker (did you mean speed.num_workers?)",
    ):
        assert expected in message
    # Nothing close: list the valid keys at that level instead of guessing.
    assert "Unknown config key frobnicate (valid keys at the top level: csv, " in message


def test_key_checks_never_evaluate_interpolations(cli, tmp_path):
    """Keys are checked before ``OmegaConf.resolve``: an unknown key whose value is an
    interpolation that cannot resolve still reports the key, not the interpolation."""
    config_path = _write_inputs(
        tmp_path, "speed:\n  num_worker: ${oc.env:HS2P_TEST_UNSET_VARIABLE}\n"
    )

    with pytest.raises(ValueError, match=r"Unknown config key speed\.num_worker "):
        cli(config_path)


def _load(config_path: Path, *overrides: str):
    """File + CLI loading, without setup()'s output-directory and logging side effects."""
    return get_cfg_from_args(
        SimpleNamespace(
            config_file=str(config_path),
            output_dir=None,
            opts=list(overrides),
            skip_datetime=True,
            skip_logging=True,
        )
    )


def test_new_labels_in_the_mask_label_maps_are_valid(tmp_path):
    config_path = _write_inputs(
        tmp_path,
        "tiling:\n"
        "  masks:\n"
        "    pixel_mapping:\n"
        "      necrosis: 2\n"
        "    colors:\n"
        "      necrosis: [255, 0, 0]\n"
        "    min_coverage:\n"
        "      necrosis: 0.5\n",
    )

    cfg = _load(config_path, "tiling.masks.pixel_mapping.stroma=3")

    assert cfg.tiling.masks.pixel_mapping.necrosis == 2
    assert cfg.tiling.masks.pixel_mapping.stroma == 3
    assert cfg.tiling.masks.min_coverage.necrosis == 0.5


def test_wandb_keys_are_not_validated(tmp_path):
    config_path = _write_inputs(tmp_path, "wandb:\n  entity: my-team\n")

    cfg = _load(config_path, "wandb.notes=first-run")

    assert cfg.wandb.entity == "my-team"
    assert cfg.wandb.notes == "first-run"


def _documented_yaml_configs() -> list[tuple[str, str]]:
    """The tracked default config plus every YAML example in the README and docs."""
    default_path = REPO_ROOT / "hs2p" / "configs" / "default.yaml"
    blocks = [(str(default_path.relative_to(REPO_ROOT)), default_path.read_text())]
    for doc in [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]:
        for index, block in enumerate(
            re.findall(r"^```yaml\n(.*?)^```", doc.read_text(), flags=re.S | re.M)
        ):
            blocks.append((f"{doc.relative_to(REPO_ROOT)}[{index}]", block))
    return blocks


@pytest.mark.parametrize(
    "block",
    [block for _, block in _documented_yaml_configs()],
    ids=[name for name, _ in _documented_yaml_configs()],
)
def test_tracked_config_examples_load(tmp_path, block):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(block)

    get_cfg_from_file(config_path)


# Programmatic callers: the resolvers check the sections they read, whatever the adapter.


def _slide2vec_adapter(**tiling_extra):
    """Shaped like slide2vec's runtime/tiling.py adapter: a partial SimpleNamespace with no
    seg_params, filter_params, preview or read_coordinates_from."""
    return SimpleNamespace(
        tiling=SimpleNamespace(
            masks=SimpleNamespace(
                output_mode="per_annotation",
                pixel_mapping={"background": 0, "tissue": 1},
                colors={"background": None, "tissue": [157, 219, 129]},
                min_coverage={"background": None, "tissue": 0.01},
            ),
            params=SimpleNamespace(
                requested_spacing_um=0.5,
                requested_tile_size_px=224,
                tolerance=0.05,
                overlap=0.0,
            ),
            independent_sampling=False,
            backend="auto",
            mask_backend="auto",
            **tiling_extra,
        )
    )


def test_resolve_tiling_config_accepts_a_partial_slide2vec_shaped_adapter():
    adapter = _slide2vec_adapter()

    tiling = resolve_tiling_config(adapter)

    assert tiling.requested_tile_size_px == 224
    assert resolve_sampling_request(adapter, tiling=tiling) == (None, None, None)


def test_resolve_tiling_config_rejects_unknown_member_on_a_simplenamespace_adapter():
    adapter = _slide2vec_adapter(on_the_fly=True)
    adapter.tiling.params.spacng = 0.5

    with pytest.raises(ValueError) as excinfo:
        resolve_tiling_config(adapter)

    message = str(excinfo.value)
    assert "Unknown config key tiling.on_the_fly (valid keys at tiling: " in message
    assert (
        "Unknown config key tiling.params.spacng "
        "(did you mean tiling.params.requested_spacing_um?)"
    ) in message


@dataclass(frozen=True)
class _SomaShapedMasks:
    """Shaped like soma's ``MasksConfig`` dataclass, passed as ``tiling.masks``."""

    pixel_mapping: dict
    min_coverage: dict = field(default_factory=dict)
    colors: dict | None = None


@dataclass(frozen=True)
class _MasksWithStrayField(_SomaShapedMasks):
    min_coverag: dict = field(default_factory=dict)


def _soma_adapter(masks):
    return SimpleNamespace(tiling=SimpleNamespace(masks=masks))


def _tiling_config():
    return resolve_tiling_config(_slide2vec_adapter())


def test_resolve_sampling_spec_accepts_a_soma_shaped_dataclass_adapter():
    masks = _SomaShapedMasks(
        pixel_mapping={"background": 0, "tumor": [1, 2]},
        min_coverage={"tumor": 0.5},
    )

    spec = resolve_sampling_spec(_soma_adapter(masks), tiling=_tiling_config())

    assert spec.active_annotations == ("tumor",)


def test_resolve_sampling_spec_rejects_unknown_member_on_a_dataclass_adapter():
    masks = _MasksWithStrayField(
        pixel_mapping={"background": 0, "tumor": 1},
        min_coverage={"tumor": 0.5},
        min_coverag={"tumor": 0.9},
    )

    with pytest.raises(ValueError) as excinfo:
        resolve_sampling_spec(_soma_adapter(masks), tiling=_tiling_config())

    assert (
        "Unknown config key tiling.masks.min_coverag "
        "(did you mean tiling.masks.min_coverage?)"
    ) in str(excinfo.value)


def test_resolve_preview_config_rejects_unknown_member():
    cfg = SimpleNamespace(
        tiling=SimpleNamespace(
            preview=SimpleNamespace(
                **dict(default_config.tiling.preview), save_mask_previw=False
            )
        )
    )

    with pytest.raises(ValueError) as excinfo:
        resolve_preview_config(cfg)

    assert (
        "Unknown config key tiling.preview.save_mask_previw "
        "(did you mean tiling.preview.save_mask_preview?)"
    ) in str(excinfo.value)


@pytest.mark.parametrize(
    ("resolve", "section", "stray_key", "suggestion"),
    [
        (resolve_segmentation_config, "seg_params", "sthres", "sthresh"),
        (resolve_filter_config, "filter_params", "filter_whit", "filter_white"),
    ],
    ids=["resolve_segmentation_config", "resolve_filter_config"],
)
def test_dataclass_resolvers_report_unknown_keys_with_the_shared_message(
    resolve, section, stray_key, suggestion
):
    cfg = OmegaConf.create(
        {"tiling": {section: {**default_config.tiling[section], stray_key: 10}}}
    )

    with pytest.raises(ValueError) as excinfo:
        resolve(cfg)

    assert (
        f"Unknown config key tiling.{section}.{stray_key} "
        f"(did you mean tiling.{section}.{suggestion}?)"
    ) in str(excinfo.value)
