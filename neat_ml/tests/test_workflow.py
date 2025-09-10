import logging
from pathlib import Path
from typing import Any, Sequence

import pytest
import os
import shutil

import neat_ml.workflow.lib_workflow as wf

def assert_logged(caplog: pytest.LogCaptureFixture, level: int, expected_message: str) -> None:
    """
    Assert a log record exists with the given level and exact message.
    """
    matched = any(rec.levelno == level and rec.getMessage() == expected_message for rec in caplog.records)
    if not matched:
        dump = "\n".join(f"[{r.levelname}] {r.getMessage()}" for r in caplog.records)
        raise AssertionError(f"Expected log at level {level} with message:\n{expected_message}\nGot:\n{dump}")

def test_get_path_structure_builds_expected_paths(tmp_path: Path):
    """
    get_path_structure: builds proc_dir and det_dir using ds_id/method/class/time_label.
    """
    roots = {"work": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}

    paths = wf.get_path_structure(roots, ds)

    base = tmp_path / "DS1" / "OpenCV" / "pos" / "T01"
    assert paths["proc_dir"] == base / "T01_Processed_OpenCV"
    assert paths["det_dir"] == base / "T01_Processed_OpenCV_With_Blob_Data"


def test_get_path_structure_missing_work_raises_keyerror(tmp_path: Path):
    """
    If 'work' key is missing in roots, a KeyError is raised.
    """
    roots = {"result": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}

    with pytest.raises(KeyError, match="work"):
        wf.get_path_structure(roots, ds)

def test_stage_opencv_warns_when_paths_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path):
    """
    stage_opencv: if 'det_dir' (or 'proc_dir') missing -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "DS3", "method": "OpenCV", "detection": {"img_dir": tmp_path}}
    paths = {"proc_dir": tmp_path / "p"}  # det_dir missing

    wf.stage_opencv(ds, paths)

    assert "Detection paths not built (step not selected or misconfig). Skipping." in caplog.text


def test_stage_opencv_warns_when_img_dir_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path):
    """
    stage_opencv: if detection.img_dir missing -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "DS4", "method": "OpenCV", "detection": {}}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}

    wf.stage_opencv(ds, paths)

    assert "No 'detection.img_dir' set for dataset 'DS4'. Skipping detection." in caplog.text


def test_stage_opencv_skips_if_output_already_exists(
    caplog: pytest.LogCaptureFixture, 
    tmp_path: Path, 
):
    """
    stage_opencv: if *_bubble_data.parquet.gzip exists -> skip and DO NOT call pipeline steps.
    """
    caplog.set_level(logging.INFO)

    proc_dir = tmp_path / "proc"
    det_dir = tmp_path / "det"
    det_dir.mkdir(parents=True)
    (det_dir / "anything_bubble_data.parquet.gzip").write_text("done")

    ds = {"id": "DS5", "method": "OpenCV", "detection": {"img_dir": str(tmp_path)}}
    paths = {"proc_dir": proc_dir, "det_dir": det_dir}

    wf.stage_opencv(ds, paths)

    assert "Detection already exists for DS5. Skipping." in caplog.text


def test_stage_opencv_pipeline_runs(
    tmpdir,
    reference_images: tuple,
):
    """
    stage_opencv: test that ``stage_opencv``
    runs successfully when provided the appropriate
    directories.
    """
    raw_image = Path(reference_images[3])
    with tmpdir.as_cwd():
        proc_dir = Path("proc_dir")
        det_dir = Path("det_dir")
        img_dir = Path("img_dir")
        tif_dir = img_dir / "tif_img"
        tiff_dir = img_dir / "tiff_img"
        tif_dir.mkdir(parents=True)
        tiff_dir.mkdir()
        # make copies of ``raw_image`` in ``img_dir``
        # with the extensions ``.tiff`` and ``.tif``
        shutil.copy(raw_image, tif_dir / "tif_img.tif")
        shutil.copy(raw_image, tiff_dir / "tiff_img.tiff")

        ds = {
            "id": "DS6", 
            "method": "OpenCV", 
            "detection": {
                "img_dir": img_dir, 
                "debug": True
            }
        }
        paths = {"proc_dir": proc_dir, "det_dir": det_dir}

        wf.stage_opencv(ds, paths)
        
        proc_dir_exp = Path(tmpdir) / proc_dir
        det_dir_exp = Path(tmpdir) / det_dir
    # assert that the ``.tiff`` and ``.tif`` images in subdirectores
    # of ``img_dir`` were preprocessed and that detection was run.
    assert (set(os.listdir(proc_dir_exp)) ==
        set(['tif_img.tif', 'tiff_img.tiff']))
    assert (set(os.listdir(det_dir_exp)) == 
        set(
            [
                'tif_img_debug.png',
                'tif_img_bubble_data.parquet.gzip',
                '.joblib_cache',
                'tiff_img_bubble_data.parquet.gzip',
                'tiff_img_debug.png'
            ]
        )
    )
    # here and above, check that the absolute file paths for ``proc_dir``
    # and ``det_dir`` are generated by ``stage_opencv``
    assert proc_dir_exp.is_absolute() 
    assert det_dir_exp.is_absolute()

def test_stage_detect_routes_to_opencv(
    tmp_path: Path,
    reference_images,
):
    """
    stage_detect: method 'OpenCV' routes to stage_opencv.
    """
    ds = {
        "id": "DS7",
        "method": "OpenCV",
        "detection": {
            "img_dir": Path(reference_images[3])
        }
    }
    paths = {
        "proc_dir": tmp_path / "p",
        "det_dir": tmp_path / "d",
    }
    exp_columns = {
        "image_filepath",
        "num_blobs_opencv",
        "median_radii_opencv"
    }
    df_out = wf.stage_detect(ds, paths)
    # assert that a detection dataframe was generated by 
    # calling the ``stage_opencv`` function
    assert df_out is not None
    assert df_out.shape == (1, 3)
    assert exp_columns.issubset(df_out.columns)


def test_stage_detect_unknown_method_error(
    tmp_path: Path
):
    """
    stage_detect: unknown method -> ValueError.
    """
    ds = {"id": "DS8", "method": "SomethingElse"}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}
    
    with pytest.raises(ValueError, match="Unknown detection method"):
        wf.stage_detect(ds, paths)


def test_stage_bubblesam_warns_when_det_dir_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_bubblesam: requires det_dir; if missing -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "BS1", "method": "BubbleSAM", "detection": {"img_dir": str(tmp_path)}}
    paths = {"proc_dir": tmp_path / "p"}

    wf.stage_bubblesam(ds, paths)

    assert_logged(caplog, logging.WARNING, "Missing detection paths (not selected or misconfigured). Skipping.")


def test_stage_bubblesam_warns_when_img_dir_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_bubblesam: if neither detection.img_dir nor dataset.img_dir provided -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "BS2", "method": "BubbleSAM", "detection": {}}
    paths = {"det_dir": tmp_path / "d"}

    wf.stage_bubblesam(ds, paths)

    assert_logged(caplog, logging.WARNING, "No detection.img_dir set for dataset 'BS2'. Skipping.")


def test_stage_bubblesam_skips_if_output_exists(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    stage_bubblesam: if *_masks_filtered.pkl exists -> skip and DO NOT call pipeline.
    """
    caplog.set_level(logging.INFO)

    det_dir = tmp_path / "det"
    det_dir.mkdir(parents=True)
    (det_dir / "foo_masks_filtered.pkl").write_text("done")

    def bad(*_: Any, **__: Any) -> None:  # pragma: no cover
        raise AssertionError("BubbleSAM pipeline should not be called when outputs exist")

    monkeypatch.setattr(wf, "collect_tiff_paths", bad)
    monkeypatch.setattr(wf, "build_df_from_img_paths", bad)
    monkeypatch.setattr(wf, "run_bubblesam", bad)

    ds = {"id": "BS3", "method": "BubbleSAM", "detection": {"img_dir": str(tmp_path)}}
    paths = {"det_dir": det_dir}

    wf.stage_bubblesam(ds, paths)

    assert_logged(caplog, logging.INFO, "BubbleSAM outputs exist for BS3. Skipping.")


def test_stage_bubblesam_happy_path_calls_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    stage_bubblesam: happy path calls collect -> build_df -> run_bubblesam using detection.img_dir.
    """
    det_dir = tmp_path / "det"
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    calls: list[str] = []

    def fake_collect(p: Path) -> Sequence[Path]:
        assert p == img_dir
        calls.append("collect")
        return [img_dir / "img1.tiff", img_dir / "img2.tiff"]

    def fake_build(paths: Sequence[Path]) -> dict[str, Any]:
        assert len(paths) == 2
        calls.append("build_df")
        return {"rows": len(paths)}

    def fake_run(df: dict[str, Any], out_dir: Path) -> None:
        assert df == {"rows": 2}
        assert out_dir == det_dir
        calls.append("run_bubblesam")

    monkeypatch.setattr(wf, "collect_tiff_paths", fake_collect)
    monkeypatch.setattr(wf, "build_df_from_img_paths", fake_build)
    monkeypatch.setattr(wf, "run_bubblesam", fake_run)

    ds = {"id": "BS4", "method": "bubblesam", "detection": {"img_dir": str(img_dir)}}
    paths = {"det_dir": det_dir}

    wf.stage_bubblesam(ds, paths)

    assert calls == ["collect", "build_df", "run_bubblesam"]
    assert det_dir.exists()


def test_stage_bubblesam_uses_dataset_level_img_dir_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    stage_bubblesam: uses dataset.img_dir when detection.img_dir is not provided.
    """
    det_dir = tmp_path / "det"
    img_dir = tmp_path / "imgs_fallback"
    img_dir.mkdir()

    used: dict[str, Path] = {}

    def fake_collect(p: Path) -> Sequence[Path]:
        used["img_dir"] = p
        return [img_dir / "im.tiff"]

    def fake_build(_: Sequence[Path]) -> dict[str, Any]:
        return {"rows": 1}

    def fake_run(_: dict[str, Any], out_dir: Path) -> None:
        used["out_dir"] = out_dir

    monkeypatch.setattr(wf, "collect_tiff_paths", fake_collect)
    monkeypatch.setattr(wf, "build_df_from_img_paths", fake_build)
    monkeypatch.setattr(wf, "run_bubblesam", fake_run)

    ds = {"id": "BS5", "method": "bubblesam", "img_dir": str(img_dir)}
    paths = {"det_dir": det_dir}

    wf.stage_bubblesam(ds, paths)

    assert used["img_dir"] == img_dir
    assert used["out_dir"] == det_dir
