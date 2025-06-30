import torch
import cv2
import argparse
import os
import numpy as np
import sam2_camera_predictor as sam2_camera
from enum import Enum
from pathlib import Path

WORKING_DIR = os.getcwd()
CONFIG_PATH = os.path.join(WORKING_DIR, "configs")
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "checkpoints")

SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

sam2p1_configs = [
    f"{SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt",
    f"{SAM2p1_BASE_URL}/sam2.1_hiera_small.pt",
    f"{SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt",
    f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"]

class MODEL_SIZE(Enum):
    TINY = 0
    SMALL = 1
    BASE_PLUS = 2
    LARGE = 3

map_to_config = {
    MODEL_SIZE.TINY: ["sam2.1_hiera_t.yaml", sam2p1_configs[MODEL_SIZE.TINY.value]],
    MODEL_SIZE.SMALL: ["sam2.1_hiera_s.yaml", sam2p1_configs[MODEL_SIZE.SMALL.value]],
    MODEL_SIZE.BASE_PLUS: ["sam2.1_hiera_base+.yaml", sam2p1_configs[MODEL_SIZE.BASE_PLUS.value]],
    MODEL_SIZE.LARGE: ["sam2.1_hiera_l.yaml", sam2p1_configs[MODEL_SIZE.LARGE.value]] }

map_to_enum = {
    "tiny" : MODEL_SIZE.TINY,
    "small" : MODEL_SIZE.SMALL,
    "base_plus" : MODEL_SIZE.BASE_PLUS,
    "large" : MODEL_SIZE.LARGE,}

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description="SAM2.1 demo")
parser.add_argument("--outdir", type=str, default=os.path.join(WORKING_DIR))
parser.add_argument("--checkpoint", type=str, default="large", choices=["tiny", "small", "base_plus", "large"])

args = parser.parse_args()

enum = map_to_enum[args.checkpoint]
link = map_to_config[enum]
path_to_yaml = os.path.join(CONFIG_PATH, link[0])
path_to_chkp = os.path.join(CHECKPOINT_PATH, Path(link[1]).name)

if not os.path.exists(CONFIG_PATH):
    print('Config path for sam2.1 does not exist, exiting...')
    
if not os.path.exists(path_to_yaml):
    print(f'Config {link[0]} for sam2.1 does not exist, exiting...')

predictor = sam2_camera.build_sam2_camera_predictor(path_to_yaml, path_to_chkp)

point = None
point_selected = False
if_init = False
random_color = True

def collect_point(event, x, y, flags, param):
    global point, point_selected
    if not point_selected and event == cv2.EVENT_LBUTTONDOWN:
        point = [x, y]
        point_selected = True

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", collect_point)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    temp_frame = frame.copy()

    if not point_selected:
        cv2.putText(temp_frame, "Select an object by clicking a point", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        if not if_init:
            if_init = True
            predictor.load_first_frame(frame)

            ann_frame_idx = 0
            ann_obj_id = (1,)
            labels = np.array([1], dtype=np.int32)
            points = np.array([point], dtype=np.float32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
            )
        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros_like(frame, dtype=np.uint8)
        
        if random_color:
            color = tuple(np.random.randint(0, 256, size=3))
            out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            #for c in range(3):
            colored_mask[:, :, 0] = out_mask[:, :, 0] * 255
            colored_mask[:, :, 1] = out_mask[:, :, 0] * 0
            colored_mask[:, :, 2] = out_mask[:, :, 0] * 0
        else:
            out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            colored_mask = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2RGB)

        all_mask = cv2.addWeighted(all_mask, 1, colored_mask, 1, 0)
        frame = cv2.addWeighted(frame, 1, all_mask, 1, 0)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()