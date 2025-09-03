import logging
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from itertools import chain

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import run_opencv
from neat_ml.bubblesam.bubblesam import run_bubblesam

__all__ = ["get_path_structure", "stage_opencv", "stage_bubblesam", "stage_detect"]

log = logging.getLogger(__name__)

def get_path_structure(
    roots: dict[str, str],
    dataset_config: dict[str, Any],
) -> dict[str, Path]:
    """
    Build only the paths needed by active steps.

    Parameters
    ----------
    roots : dict[str, str]
        Root dirs (work).
    dataset_config : dict[str, Any]
        Dataset dict (id, method, class, time_label, detection).

    Returns
    -------
    paths : dict[str, Path]
        Paths keyed by step usage (proc_dir, det_dir).
    """
    paths = {}
    ds_id = dataset_config.get("id", "unknown")
    method = dataset_config.get("method", "")
    class_label = dataset_config.get("class", "")
    time_label = dataset_config.get("time_label", "")
    work_root = Path(roots["work"])

    base_proc: Path = work_root / ds_id / method / class_label / time_label

    if method == 'OpenCV':
        paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"

    paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    return paths

def stage_opencv(
    dataset_config: dict[str, Any],
    paths: dict[str, Path]
) -> Optional[pd.DataFrame]:
    """
    Run OpenCV preprocessing + detection when configured.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config. Expects 'method' == 'OpenCV' and 'detection' block.
    paths : dict[str, Path]
        Paths from get_path_structure() (proc_dir, det_dir if built).
    
    Returns:
    --------
    df_out: Optional[pd.DataFrame]
        dataframe containing summary of opencv bubble detection
        information 
    """
    detection_cfg = dataset_config.get("detection", {})
    img_dir_str = detection_cfg.get("img_dir")
    debug = detection_cfg.get("debug", False)
    ds_id = dataset_config.get("id", "unknown")
    if "proc_dir" not in paths or "det_dir" not in paths:
        log.warning("Detection paths not built (step not selected or misconfig). Skipping.")
        return None
    if not img_dir_str:
        log.warning(f"No 'detection.img_dir' set for dataset '{ds_id}'. Skipping detection.")
        return None
    
    proc_dir = paths["proc_dir"]
    det_dir = paths["det_dir"]
    img_dir = Path(img_dir_str).expanduser().resolve()

    if list(det_dir.glob("*_bubble_data.parquet.gzip")):
        log.info(f"Detection already exists for {ds_id}. Skipping.")
        return None
    
    # expand user inputs to absolute file paths  
    proc_dir = proc_dir.expanduser().resolve()
    det_dir = det_dir.expanduser().resolve()
    proc_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Preprocessing (OpenCV) for {ds_id} -> {proc_dir}")
    cv_preprocess(img_dir, proc_dir)
    
    log.info(f"Detecting (OpenCV) for {ds_id} -> {det_dir}")
    # collect paths for preprocessed tiff image files, store in DataFrame
    img_paths = chain(proc_dir.glob("*.tiff"), proc_dir.glob("*.tif"))
    df_imgs = pd.DataFrame({"image_filepath": img_paths})
    df_out = run_opencv(df_imgs, det_dir, debug=debug)
    log.info("OpenCV Detection Ran Successfully.")

    return df_out


def stage_bubblesam(dataset_config: dict[str, Any], paths: dict[str, Path]) -> None:
    """
    Run BubbleSAM detection when method='BubbleSAM'.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config. Expects method 'BubbleSAM'.
        Uses detection.img_dir (falls back to dataset.img_dir).
    paths : dict[str, Path]
        Must include proc_dir and det_dir.

    Returns
    -------
    None
        Writes preprocessed images and *_masks_filtered.pkl.
    """
    if "det_dir" not in paths:
        log.warning("Missing detection paths (not selected or misconfigured). Skipping.")
        return

    det_cfg = dict(dataset_config.get("detection", {}))
    img_dir_str: Optional[str] = det_cfg.get("img_dir", dataset_config.get("img_dir"))
    if not img_dir_str:
        log.warning("No detection.img_dir set for dataset '%s'. Skipping.", dataset_config.get("id"))
        return

    ds_id: str = str(dataset_config.get("id", "unknown"))
    det_dir: Path = paths["det_dir"]
    img_dir: Path = Path(img_dir_str)

    if list(det_dir.glob("*_masks_filtered.pkl")):
        log.info("BubbleSAM outputs exist for %s. Skipping.", ds_id)
        return

    det_dir.mkdir(parents=True, exist_ok=True)
    log.info("Detecting (BubbleSAM) for %s -> %s", ds_id, det_dir)
    # collect paths for preprocessed tiff image files, store in DataFrame
    img_paths = chain(img_dir.glob("*.tiff"), img_dir.glob("*.tif"))
    df_imgs = pd.DataFrame({"image_filepath": img_paths})
    run_bubblesam(df_imgs, det_dir)


def stage_detect(
    dataset_config: dict[str, Any],
    paths: dict[str, Path]
) -> Optional[pd.DataFrame]:
    """
    Route detection to OpenCV or BubbleSAM based on dataset.method.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config with 'method'.
    paths : dict[str, Path]
        Detection paths (proc_dir, det_dir).

    Returns:
    --------
    df_out: Optional[pd.DataFrame]
        dataframe containing summary of opencv bubble detection
        information 
    """
    method = dataset_config.get("method", "").lower()
    ds_id = dataset_config.get("id")
    if method == "opencv":
        df_out = stage_opencv(dataset_config, paths)
    elif method == "bubblesam":
        stage_bubblesam(dataset_config, paths)
    else:
        raise ValueError(f"Unknown detection method '{method}' for dataset '{ds_id}'.")
    return df_out 
