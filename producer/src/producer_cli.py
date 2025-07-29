import os
import requests
import shutil
import argparse

from enum import Enum
from pathlib import Path
from tqdm import tqdm

WORKING_DIR = os.getcwd()
CONFIG_PATH = os.path.join(WORKING_DIR, "configs")
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "checkpoints")

SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824"

sam2p1_configs = [
    f"{SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt",
    f"{SAM2p1_BASE_URL}/sam2.1_hiera_small.pt",
    f"{SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt",
    f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"]

sam2_configs = [
    f"{SAM2_BASE_URL}/sam2_hiera_tiny.pt",
    f"{SAM2_BASE_URL}/sam2_hiera_small.pt",
    f"{SAM2_BASE_URL}/sam2_hiera_base_plus.pt",
    f"{SAM2_BASE_URL}/sam2_hiera_large.pt"]

class MODEL_SIZE(Enum):
    TINY = 0
    SMALL = 1
    BASE_PLUS = 2
    LARGE = 3

map_to_config = {
    MODEL_SIZE.TINY: [os.path.join("sam2.1", "sam2.1_hiera_t.yaml"), sam2p1_configs[MODEL_SIZE.TINY.value]],
    MODEL_SIZE.SMALL: [os.path.join("sam2.1", "sam2.1_hiera_s.yaml"), sam2p1_configs[MODEL_SIZE.SMALL.value]],
    MODEL_SIZE.BASE_PLUS: [os.path.join("sam2.1", "sam2.1_hiera_base+.yaml"), sam2p1_configs[MODEL_SIZE.BASE_PLUS.value]],
    MODEL_SIZE.LARGE: [os.path.join("sam2.1", "sam2.1_hiera_l.yaml"), sam2p1_configs[MODEL_SIZE.LARGE.value]] }

map_to_config_sam2 = {
    MODEL_SIZE.TINY: [os.path.join("sam2", "sam2_hiera_t.yaml"), sam2_configs[MODEL_SIZE.TINY.value]],
    MODEL_SIZE.SMALL: [os.path.join("sam2", "sam2_hiera_s.yaml"), sam2_configs[MODEL_SIZE.SMALL.value]],
    MODEL_SIZE.BASE_PLUS: [os.path.join("sam2", "sam2_hiera_b+.yaml"), sam2_configs[MODEL_SIZE.BASE_PLUS.value]],
    MODEL_SIZE.LARGE: [os.path.join("sam2", "sam2_hiera_l.yaml"), sam2_configs[MODEL_SIZE.LARGE.value]] }

map_to_enum = {
    "tiny" : MODEL_SIZE.TINY,
    "small" : MODEL_SIZE.SMALL,
    "base_plus" : MODEL_SIZE.BASE_PLUS,
    "large" : MODEL_SIZE.LARGE,}

producer_cli = argparse.ArgumentParser(description="HOPE producer")
producer_cli.add_argument("--sam2_checkpoint", type=str, default="large", choices=["tiny", "small", "base_plus", "large"])
producer_cli.add_argument("--sam2_image_size", type=int, default=1024, choices=[1024, 512, 256, 128])
producer_cli.add_argument("--yolo_size", type=str, default="large", choices=["samll, medium, large"])
producer_cli.add_argument("--realsense_clr_capture_width", type=int, default=848, choices=[848])
producer_cli.add_argument("--realsense_clr_capture_height", type=int, default=480, choices=[480])
producer_cli.add_argument("--realsense_depth_capture_width", type=int, default=848, choices=[848])
producer_cli.add_argument("--realsense_depth_capture_height", type=int, default=480, choices=[480])
producer_cli.add_argument("--realsense_target_fps", type=int, default=30, choices=[30])
producer_cli.add_argument("--cluster_predictor", type=str, default="sam2", choices=["yolo", "sam2",])

def getRequest(outputPath : Path, url : str) -> None:
    req = requests.get(url, stream=True, allow_redirects=True, timeout=10)

    if req.status_code != 200:
        print('URL: {url}, does not exist.'\
              'You have to download it and extract it manually')
        return

    fileSize = int(req.headers.get('Content-Length', 0))
    desc = "(Unknown total file size)" if fileSize == 0 else ""
    fileName = req.headers.get("Content-Disposition")
    if fileName is not None:
        fileName = fileName.split("filename=")[1].replace("\"", "").replace(";", "")
    else : #more like a hack some day this will fail.
        fileName = Path(url).name

    print(f'Downloading: {fileName}')
    outputFile = Path(outputPath, fileName)

    with tqdm.wrapattr(req.raw, "read", total=fileSize, desc=desc) as r_raw:
        with open(outputFile, "wb") as f:
            shutil.copyfileobj(r_raw, f)