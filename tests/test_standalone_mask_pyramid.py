from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import csv

import numpy as np
import openslide
import pytest
import tifffile

pytestmark = pytest.mark.script


def _load_script_module():
	script_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_tissue_mask_pyramid.py"
	spec = spec_from_file_location("generate_tissue_mask_pyramid", script_path)
	assert spec is not None and spec.loader is not None
	module = module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def test_write_pyramidal_mask_tiff_uses_top_level_reduced_ifds(tmp_path):
	script = _load_script_module()

	level0 = np.zeros((64, 64), dtype=np.uint8)
	level1 = np.zeros((32, 32), dtype=np.uint8)
	level2 = np.zeros((16, 16), dtype=np.uint8)
	output_path = tmp_path / "mask-pyramid.tif"

	script.write_pyramidal_mask_tiff(
		levels=[level0, level1, level2],
		output_path=output_path,
		level0_spacing=1.0,
		downsample_per_level=2.0,
		compression="deflate",
		tile_size=16,
	)

	with tifffile.TiffFile(output_path) as tf:
		assert len(tf.pages) == 3
		assert tf.pages[0].shape == (64, 64)
		assert tf.pages[1].shape == (32, 32)
		assert tf.pages[2].shape == (16, 16)
		assert tf.pages[1].tags["NewSubfileType"].value == tifffile.FILETYPE.REDUCEDIMAGE
		assert tf.pages[2].tags["NewSubfileType"].value == tifffile.FILETYPE.REDUCEDIMAGE


def test_write_pyramidal_mask_tiff_is_multilevel_for_openslide(tmp_path):
	script = _load_script_module()

	level0 = np.ones((64, 64), dtype=np.uint8)
	level1 = np.ones((32, 32), dtype=np.uint8)
	level2 = np.ones((16, 16), dtype=np.uint8)
	output_path = tmp_path / "mask-pyramid-openslide.tif"

	script.write_pyramidal_mask_tiff(
		levels=[level0, level1, level2],
		output_path=output_path,
		level0_spacing=1.0,
		downsample_per_level=2.0,
		compression="deflate",
		tile_size=16,
	)

	slide = openslide.OpenSlide(str(output_path))
	assert slide.level_count == 3
	assert slide.level_dimensions == ((64, 64), (32, 32), (16, 16))


def test_resolve_wsi_paths_supports_single_and_glob(tmp_path):
	script = _load_script_module()

	slide_a = tmp_path / "slide-a.tif"
	slide_b = tmp_path / "slide-b.tif"
	not_slide = tmp_path / "notes.txt"
	slide_a.touch()
	slide_b.touch()
	not_slide.touch()

	resolved = script.resolve_wsi_paths([str(slide_a), str(tmp_path / "*.tif")])
	assert resolved == [slide_a, slide_b]


def test_build_output_mapping_single_mode_with_output_dir(tmp_path):
	script = _load_script_module()

	slide = tmp_path / "one-slide.tif"
	slide.touch()
	output_dir = tmp_path / "out"

	mapping = script.build_output_mapping(
		input_wsi_paths=[slide],
		output_path=None,
		output_dir=output_dir,
	)

	assert mapping == [(slide, output_dir / "one-slide.tif")]


def test_build_output_mapping_multi_mode_requires_output_dir(tmp_path):
	script = _load_script_module()

	slide_a = tmp_path / "slide-a.tif"
	slide_b = tmp_path / "slide-b.tif"
	slide_a.touch()
	slide_b.touch()

	with pytest.raises(ValueError, match="--output-dir"):
		script.build_output_mapping(
			input_wsi_paths=[slide_a, slide_b],
			output_path=tmp_path / "single.tif",
			output_dir=None,
		)


def test_process_slides_continues_after_failure(tmp_path, monkeypatch):
	script = _load_script_module()

	slide_ok = tmp_path / "ok.tif"
	slide_fail = tmp_path / "fail.tif"
	out_ok = tmp_path / "out" / "ok.tif"
	out_fail = tmp_path / "out" / "fail.tif"
	slide_ok.touch()
	slide_fail.touch()

	def fake_load_wsi_at_spacing(**kwargs):
		if kwargs["wsi_path"].name == "fail.tif":
			raise ValueError("spacing mismatch")
		return np.zeros((16, 16, 3), dtype=np.uint8), 1.0

	def fake_segment_tissue_hsv(**kwargs):
		return np.zeros((16, 16), dtype=np.uint8)

	def fake_build_mask_pyramid(**kwargs):
		return [np.zeros((16, 16), dtype=np.uint8)]

	def fake_write_pyramidal_mask_tiff(**kwargs):
		kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
		kwargs["output_path"].touch()

	monkeypatch.setattr(script, "load_wsi_at_spacing", fake_load_wsi_at_spacing)
	monkeypatch.setattr(script, "segment_tissue_hsv", fake_segment_tissue_hsv)
	monkeypatch.setattr(script, "build_mask_pyramid", fake_build_mask_pyramid)
	monkeypatch.setattr(script, "write_pyramidal_mask_tiff", fake_write_pyramidal_mask_tiff)

	results = script.process_slides(
		output_mapping=[(slide_ok, out_ok), (slide_fail, out_fail)],
		spacing=1.0,
		tolerance=0.1,
		backend="openslide",
		spacing_at_level_0=None,
		downsample_per_level=2.0,
		min_size=8,
		compression="deflate",
		tile_size=16,
		verbose=False,
		command_signature="sig-1",
		cache_manifest_path=tmp_path / "out" / "cache_manifest.json",
	)

	assert [row["status"] for row in results] == ["success", "failed"]
	assert "spacing mismatch" in results[1]["traceback"]
	assert out_ok.is_file()
	assert not out_fail.is_file()


def test_write_summary_csv_contains_status_and_traceback(tmp_path):
	script = _load_script_module()
	summary_path = tmp_path / "summary.csv"
	results = [
		{
			"slide_path": "/slides/a.tif",
			"output_path": "/out/a.tif",
			"status": "success",
			"traceback": "",
		},
		{
			"slide_path": "/slides/b.tif",
			"output_path": "/out/b.tif",
			"status": "failed",
			"traceback": "ValueError: spacing mismatch",
		},
	]

	script.write_summary_csv(summary_path, results)

	with summary_path.open("r", newline="", encoding="utf-8") as f:
		rows = list(csv.DictReader(f))

	assert [row["status"] for row in rows] == ["success", "failed"]
	assert rows[0]["traceback"] == ""
	assert "spacing mismatch" in rows[1]["traceback"]


def test_process_slides_skips_when_cache_manifest_matches(tmp_path, monkeypatch):
	script = _load_script_module()

	slide = tmp_path / "slide.tif"
	output = tmp_path / "out" / "slide.tif"
	cache_manifest_path = tmp_path / "out" / "cache_manifest.json"
	slide.touch()
	output.parent.mkdir(parents=True, exist_ok=True)
	tifffile.imwrite(output, np.zeros((8, 8), dtype=np.uint8))

	input_fp = script.get_file_fingerprint(slide)
	output_fp = script.get_file_fingerprint(output)
	command_signature = "sig-1"
	script.save_cache_manifest(
		cache_manifest_path,
		command_signature=command_signature,
		entries={
			str(slide.resolve()): {
				"slide_path": str(slide.resolve()),
				"output_path": str(output.resolve()),
				"input_fingerprint": input_fp,
				"output_fingerprint": output_fp,
			}
		},
	)

	def fail_if_called(**kwargs):
		raise AssertionError("processing function should not be called on cache hit")

	monkeypatch.setattr(script, "load_wsi_at_spacing", fail_if_called)
	monkeypatch.setattr(script, "segment_tissue_hsv", fail_if_called)
	monkeypatch.setattr(script, "build_mask_pyramid", fail_if_called)
	monkeypatch.setattr(script, "write_pyramidal_mask_tiff", fail_if_called)

	results = script.process_slides(
		output_mapping=[(slide, output)],
		spacing=1.0,
		tolerance=0.1,
		backend="openslide",
		spacing_at_level_0=None,
		downsample_per_level=2.0,
		min_size=8,
		compression="deflate",
		tile_size=16,
		verbose=False,
		command_signature=command_signature,
		cache_manifest_path=cache_manifest_path,
	)

	assert [row["status"] for row in results] == ["skipped"]
	assert results[0]["traceback"] == ""


def test_process_slides_recomputes_when_output_fingerprint_changed(tmp_path, monkeypatch):
	script = _load_script_module()

	slide = tmp_path / "slide.tif"
	output = tmp_path / "out" / "slide.tif"
	cache_manifest_path = tmp_path / "out" / "cache_manifest.json"
	slide.touch()
	output.parent.mkdir(parents=True, exist_ok=True)
	output.touch()

	input_fp = script.get_file_fingerprint(slide)
	output_fp = script.get_file_fingerprint(output)
	command_signature = "sig-1"
	script.save_cache_manifest(
		cache_manifest_path,
		command_signature=command_signature,
		entries={
			str(slide.resolve()): {
				"slide_path": str(slide.resolve()),
				"output_path": str(output.resolve()),
				"input_fingerprint": input_fp,
				"output_fingerprint": output_fp,
			}
		},
	)

	output.write_bytes(b"changed")

	call_counter = {"count": 0}

	def fake_load_wsi_at_spacing(**kwargs):
		return np.zeros((16, 16, 3), dtype=np.uint8), 1.0

	def fake_segment_tissue_hsv(**kwargs):
		return np.zeros((16, 16), dtype=np.uint8)

	def fake_build_mask_pyramid(**kwargs):
		return [np.zeros((16, 16), dtype=np.uint8)]

	def fake_write_pyramidal_mask_tiff(**kwargs):
		call_counter["count"] += 1
		kwargs["output_path"].write_bytes(b"recomputed")

	monkeypatch.setattr(script, "load_wsi_at_spacing", fake_load_wsi_at_spacing)
	monkeypatch.setattr(script, "segment_tissue_hsv", fake_segment_tissue_hsv)
	monkeypatch.setattr(script, "build_mask_pyramid", fake_build_mask_pyramid)
	monkeypatch.setattr(script, "write_pyramidal_mask_tiff", fake_write_pyramidal_mask_tiff)

	results = script.process_slides(
		output_mapping=[(slide, output)],
		spacing=1.0,
		tolerance=0.1,
		backend="openslide",
		spacing_at_level_0=None,
		downsample_per_level=2.0,
		min_size=8,
		compression="deflate",
		tile_size=16,
		verbose=False,
		command_signature=command_signature,
		cache_manifest_path=cache_manifest_path,
	)

	assert [row["status"] for row in results] == ["success"]
	assert call_counter["count"] == 1


def test_process_slides_no_cache_forces_recompute(tmp_path, monkeypatch):
	script = _load_script_module()

	slide = tmp_path / "slide.tif"
	output = tmp_path / "out" / "slide.tif"
	cache_manifest_path = tmp_path / "out" / "cache_manifest.json"
	slide.touch()
	output.parent.mkdir(parents=True, exist_ok=True)
	tifffile.imwrite(output, np.zeros((8, 8), dtype=np.uint8))

	input_fp = script.get_file_fingerprint(slide)
	output_fp = script.get_file_fingerprint(output)
	command_signature = "sig-1"
	script.save_cache_manifest(
		cache_manifest_path,
		command_signature=command_signature,
		entries={
			str(slide.resolve()): {
				"slide_path": str(slide.resolve()),
				"output_path": str(output.resolve()),
				"input_fingerprint": input_fp,
				"output_fingerprint": output_fp,
			}
		},
	)

	call_counter = {"count": 0}

	def fake_load_wsi_at_spacing(**kwargs):
		return np.zeros((16, 16, 3), dtype=np.uint8), 1.0

	def fake_segment_tissue_hsv(**kwargs):
		return np.zeros((16, 16), dtype=np.uint8)

	def fake_build_mask_pyramid(**kwargs):
		return [np.zeros((16, 16), dtype=np.uint8)]

	def fake_write_pyramidal_mask_tiff(**kwargs):
		call_counter["count"] += 1
		kwargs["output_path"].write_bytes(b"forced-recompute")

	monkeypatch.setattr(script, "load_wsi_at_spacing", fake_load_wsi_at_spacing)
	monkeypatch.setattr(script, "segment_tissue_hsv", fake_segment_tissue_hsv)
	monkeypatch.setattr(script, "build_mask_pyramid", fake_build_mask_pyramid)
	monkeypatch.setattr(script, "write_pyramidal_mask_tiff", fake_write_pyramidal_mask_tiff)

	results = script.process_slides(
		output_mapping=[(slide, output)],
		spacing=1.0,
		tolerance=0.1,
		backend="openslide",
		spacing_at_level_0=None,
		downsample_per_level=2.0,
		min_size=8,
		compression="deflate",
		tile_size=16,
		verbose=False,
		command_signature=command_signature,
		cache_manifest_path=cache_manifest_path,
		no_cache=True,
	)

	assert [row["status"] for row in results] == ["success"]
	assert call_counter["count"] == 1


def test_postprocess_mask_removes_small_components_from_um2_threshold():
	script = _load_script_module()

	mask = np.zeros((20, 20), dtype=np.uint8)
	mask[2:4, 2:4] = 1      # area 4 px
	mask[10:14, 10:14] = 1  # area 16 px

	processed = script.postprocess_mask(
		mask=mask,
		spacing_um_per_px=1.0,
		min_component_area_um2=10.0,
	)

	assert processed[2:4, 2:4].sum() == 0
	assert processed[10:14, 10:14].sum() == 16


def test_postprocess_mask_fills_small_holes_not_border_background():
	script = _load_script_module()

	mask = np.ones((20, 20), dtype=np.uint8)
	mask[8:10, 8:10] = 0   # internal hole area 4 px
	mask[0:2, 0:2] = 0     # border-connected background should remain

	processed = script.postprocess_mask(
		mask=mask,
		spacing_um_per_px=1.0,
		max_hole_area_um2=10.0,
	)

	assert processed[8:10, 8:10].sum() == 4
	assert processed[0:2, 0:2].sum() == 0


def test_postprocess_mask_radius_um_scales_with_spacing():
	script = _load_script_module()

	mask = np.zeros((40, 40), dtype=np.uint8)
	mask[18:22, 18:22] = 1

	processed_fine = script.postprocess_mask(
		mask=mask,
		spacing_um_per_px=0.5,
		close_radius_um=2.0,
	)
	processed_coarse = script.postprocess_mask(
		mask=mask,
		spacing_um_per_px=2.0,
		close_radius_um=2.0,
	)

	assert processed_fine.sum() >= processed_coarse.sum()


def test_segment_tissue_hsv_uses_gaussian_blur_when_sigma_positive(monkeypatch):
	script = _load_script_module()

	call_counter = {"count": 0}

	def fake_gaussian_blur(img, ksize, sigmaX, sigmaY):
		call_counter["count"] += 1
		return img

	monkeypatch.setattr(script.cv2, "GaussianBlur", fake_gaussian_blur)

	wsi_arr = np.zeros((8, 8, 3), dtype=np.uint8)
	script.segment_tissue_hsv(wsi_arr=wsi_arr, gaussian_sigma_px=0.0)
	script.segment_tissue_hsv(wsi_arr=wsi_arr, gaussian_sigma_px=1.5)

	assert call_counter["count"] == 1
