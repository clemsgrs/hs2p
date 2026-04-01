from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import hs2p.wsi.wsi as wsi_mod
from hs2p.wsi.backends.common import make_white_canvas


class _FakeReader:
    def __init__(
        self,
        *,
        levels: list[np.ndarray],
        spacings: list[float],
        backend_name: str = "asap",
    ):
        self._levels = [np.asarray(level) for level in levels]
        self._backend_name = backend_name
        self.dimensions = (int(self._levels[0].shape[1]), int(self._levels[0].shape[0]))
        self.spacing = float(spacings[0])
        self.spacings = [float(value) for value in spacings]
        self.level_dimensions = [
            (int(level.shape[1]), int(level.shape[0])) for level in self._levels
        ]
        self.level_downsamples = [
            (
                self.level_dimensions[0][0] / float(width),
                self.level_dimensions[0][1] / float(height),
            )
            for width, height in self.level_dimensions
        ]
        self.level_count = len(self._levels)

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def read_level(self, level: int) -> np.ndarray:
        return np.asarray(self._levels[level])

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        downsample = float(self.level_downsamples[level][0])
        arr = self._levels[level]
        x_level = int(round(location[0] / downsample))
        y_level = int(round(location[1] / downsample))
        width = int(size[0])
        height = int(size[1])
        patch = make_white_canvas(width, height)
        src = arr[y_level : y_level + height, x_level : x_level + width, :]
        patch[: src.shape[0], : src.shape[1], :] = src
        return patch

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        del size
        return self.read_level(self.level_count - 1)

    def close(self) -> None:
        return None


def _sampling_spec() -> wsi_mod.SamplingSpec:
    return wsi_mod.SamplingSpec(
        pixel_mapping={"background": 0, "tumor": 9},
        color_mapping={"background": None, "tumor": None},
        tissue_percentage={"background": None, "tumor": 0.0},
        active_annotations=("tumor",),
    )


def _segment_params() -> SimpleNamespace:
    return SimpleNamespace(
        downsample=1,
        sthresh_up=255,
    )


def test_load_segmentation_aligns_mask_reader_to_slide_resolution(monkeypatch, tmp_path):
    slide_level0 = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_level0 = np.zeros((4, 4, 1), dtype=np.uint8)
    mask_level0[1:3, 1:3, 0] = 9

    slide_reader = _FakeReader(levels=[slide_level0], spacings=[1.0])
    mask_reader = _FakeReader(levels=[mask_level0], spacings=[2.0])

    mask_path = tmp_path / "aligned-mask.tif"
    Image.fromarray(mask_level0[:, :, 0]).save(mask_path)

    calls: list[Path] = []

    def _fake_open_slide(path, backend: str, *, spacing_override=None, gpu_decode=False):
        del backend, spacing_override, gpu_decode
        calls.append(Path(path))
        if "mask" in Path(path).name:
            return mask_reader
        return slide_reader

    monkeypatch.setattr(wsi_mod, "resolve_backend", lambda *args, **kwargs: SimpleNamespace(backend="asap"))
    monkeypatch.setattr(wsi_mod, "open_slide", _fake_open_slide)

    wsi = wsi_mod.WSI(
        path=tmp_path / "slide.tif",
        mask_path=mask_path,
        backend="auto",
        sampling_spec=_sampling_spec(),
        segment_params=_segment_params(),
    )

    assert calls == [tmp_path / "slide.tif", mask_path]
    assert wsi.annotation_mask["tumor"].shape == (8, 8)
    assert np.all(wsi.annotation_mask["tumor"][2:6, 2:6] == 255)
    assert np.all(wsi.annotation_mask["tumor"][:2, :] == 0)


def test_load_segmentation_falls_back_to_pil_for_discrete_labels(monkeypatch, tmp_path):
    slide_level0 = np.zeros((2, 2, 3), dtype=np.uint8)
    interpolated_mask = np.array(
        [
            [[0], [4]],
            [[5], [9]],
        ],
        dtype=np.uint8,
    )
    stored_mask = np.array(
        [
            [0, 9],
            [9, 0],
        ],
        dtype=np.uint8,
    )

    slide_reader = _FakeReader(levels=[slide_level0], spacings=[1.0])
    mask_reader = _FakeReader(levels=[interpolated_mask], spacings=[1.0])

    mask_path = tmp_path / "discrete-mask.tif"
    Image.fromarray(stored_mask).save(mask_path)

    def _fake_open_slide(path, backend: str, *, spacing_override=None, gpu_decode=False):
        del backend, spacing_override, gpu_decode
        if "mask" in Path(path).name:
            return mask_reader
        return slide_reader

    monkeypatch.setattr(wsi_mod, "resolve_backend", lambda *args, **kwargs: SimpleNamespace(backend="asap"))
    monkeypatch.setattr(wsi_mod, "open_slide", _fake_open_slide)

    wsi = wsi_mod.WSI(
        path=tmp_path / "slide.tif",
        mask_path=mask_path,
        backend="auto",
        sampling_spec=_sampling_spec(),
        segment_params=_segment_params(),
    )

    np.testing.assert_array_equal(
        wsi.annotation_mask["tumor"],
        np.array(
            [
                [0, 255],
                [255, 0],
            ],
            dtype=np.uint8,
        ),
    )
