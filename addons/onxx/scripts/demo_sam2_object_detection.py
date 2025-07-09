import os
import numpy as np
import mediapipe as mp
import cv2 as cv
import time
import sam2_config
import torch
import argparse
import matplotlib.pyplot as plt

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from pathlib import Path

def bake_anns(anns):
    if len(anns) == 0:
        return None
    sorted_anns = anns#sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    #img[:, :, 2] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        img[m, :] = np.random.random(3) 

    return (img * 255).astype(np.uint8)

def launch_demo(path_to_yaml,  path_to_chkp, device, image_size):
    config_name = Path(path_to_yaml).name
    config_path = "configs"

    sam2 = build_sam2(path_to_yaml, path_to_chkp, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2, points_per_side=16, points_per_batch=512,)
    '''
        points_per_side=2,
        points_per_batch=512,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=0,
        use_m2m=False,
        multimask_output=False)
    '''

    cap = cv.VideoCapture(0)
    cv.namedWindow("Camera")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        now = time.time()
        masks = mask_generator.generate(frame)
        clusters = bake_anns(masks)
        if clusters is not None:
            frame = cv.addWeighted(frame, 0.5, clusters, 0.5, 0)

        dt = time.time() - now

        cv.setWindowTitle("Camera", f"{1.0/(dt + 1.e-5):.2}FPS ({dt*1000:.2}ms)")
        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()

def main() :
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Realsense SAM2.1 demo")
    parser.add_argument("--checkpoint", type=str, default="large", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--image_size", type=int, default=1024)

    args = parser.parse_args()

    enum = sam2_config.map_to_enum[args.checkpoint]
    link = sam2_config.map_to_config[enum]
    path_to_yaml = os.path.join(sam2_config.CONFIG_PATH, link[0])
    print(f"Received path_to_yaml: {path_to_yaml}")
    path_to_chkp = os.path.join(sam2_config.CHECKPOINT_PATH, Path(link[1]).name)

    if args.image_size % 32 != 0:
        print(f'Requested image size {args.image_size} is not a multple of 32 falling back to SAM2.1 default 1024')
        args.image_size = 1024

    if not os.path.exists(sam2_config.CONFIG_PATH):
        print('Config path for sam2.1 does not exist, exiting...')
        return
    
    if not os.path.exists(path_to_yaml):
        print(f'Config {link[0]} for sam2.1 does not exist, You need to download them from https://github.com/facebookresearch/sam2/tree/main/sam2/configs/sam2.1, exiting...')
        return
    
    if not os.path.exists(path_to_chkp):
        print(f'Checkpoint {path_to_chkp} is missing, downloading...')
        os.makedirs(sam2_config.CHECKPOINT_PATH, exist_ok=True)
        sam2_config.getRequest(sam2_config.CHECKPOINT_PATH, link[1])

    with torch.autocast(device_type=DEVICE.__str__(), dtype=torch.bfloat16):
        launch_demo(
            path_to_yaml=path_to_yaml, 
            path_to_chkp=path_to_chkp, 
            device=DEVICE,
            image_size=args.image_size
        )

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Compiled CUDA version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    main()