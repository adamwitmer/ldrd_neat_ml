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

def run_detection(
    dataset_config: dict[str, Any],
    paths: dict[str, Path],
) -> Optional[pd.DataFrame]:
    """
    Run OpenCV preprocessing + detection or BubbleSAM detection when configured.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config. Expects 'method' == 'OpenCV' and 'detection' block OR
        ``method == BubbleSAM``
    paths : dict[str, Path]
        Paths from get_path_structure() (proc_dir, det_dir if built).
    """
    # get method (``opencv`` or ``bubblesam``) and initialize
    # variables to guide function calls
    method = dataset_config.get("method", "")
    if method.lower() == "opencv":
        check_dirs = set(["det_dir", "proc_dir"])
        file_suffix = "_bubble_data"
    else:
        check_dirs = set(["det_dir"])
        file_suffix = "_masks_filtered"
    
    # check if the appropriate image filepaths are available
    if not set(paths.keys()) == check_dirs:
        log.warning("Detection paths not built (step not selected or misconfig). Skipping.")
        return None
    
    # check if the input image filepaths data structure contains the appropriate
    # keys for performing detection
    det_dir: Path = paths["det_dir"]
    detection_cfg: dict[str, Any] = dict(dataset_config.get("detection", {}))
    img_dir_str: Optional[str] = detection_cfg.get("img_dir", dataset_config.get("img_dir"))
    if not img_dir_str:
        log.warning("No 'detection.img_dir' set for dataset '%s'. Skipping detection.",
                    dataset_config.get("id"))
        return None
    
    # check if the detection step has already been performed
    img_dir: Path = Path(img_dir_str)
    det_dir.mkdir(parents=True, exist_ok=True)
    ds_id: str = str(dataset_config.get("id", "unknown"))
    if list(det_dir.glob(f"*{file_suffix}.pkl")):
        log.info("Detection already exists for %s. Skipping.", ds_id)
        return None
    
    # for the ``opencv`` method, perform image preprocessing
    if method.lower() == "opencv":
        debug: bool = bool(detection_cfg.get("debug", False))
        tiff_paths: Path = paths["proc_dir"]
        tiff_paths.mkdir(parents=True, exist_ok=True)
        log.info("Preprocessing (OpenCV) for %s -> %s", ds_id, tiff_paths)
        cv_preprocess(img_dir, tiff_paths)
    else:
        tiff_paths = img_dir
    
    # route the detection step to the appropriate method
    log.info(f"Detecting ({method}) for %s -> %s", ds_id, det_dir)
    # collect paths for preprocessed tiff image files, store in DataFrame
    # check if the path is a single file or a directory
    if tiff_paths.is_file():
        img_paths = tiff_paths
        df_imgs = pd.DataFrame({"image_filepath": [img_paths]})
    elif tiff_paths.is_dir():
        img_paths = chain(tiff_paths.glob("*.tiff"),
            tiff_paths.glob("*.tif")
        )  # type: ignore[assignment]
        df_imgs = pd.DataFrame({"image_filepath": img_paths})
    # run specified detection method
    if method.lower() == "opencv":
        df_out = run_opencv(df_imgs, det_dir, debug=debug)
    else:
        df_out = run_bubblesam(df_imgs, det_dir)
    return df_out

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
    method: str = str(dataset_config.get("method", "")).lower()
    if method in ["opencv", "bubblesam"]:
        df_out = run_detection(dataset_config, paths)
        return df_out
    else:
        log.warning("Unknown detection method '%s' for dataset '%s'.",
                    method, dataset_config.get("id"))
        return None
