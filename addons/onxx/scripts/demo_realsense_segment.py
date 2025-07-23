import cv2
import argparse
import sam2_config
import torch
import os
import time

import pyrealsense2 as rs
import numpy as np
import sam2_camera_predictor as sam2_camera

from pathlib import Path

def launch_demo(path_to_yaml: str,  path_to_chkp: str, device: str, image_size: int):
    
    config_name = Path(path_to_yaml).name
    config_path = str(Path(".", "configs", "sam2.1"))
    predictor = sam2_camera.build_sam2_camera_predictor(
        config_file=config_name, 
        config_path=config_path,
        ckpt_path=path_to_chkp, 
        device=device,
        image_size=image_size
    )

    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    point = None
    point_selected = False
    if_init = False

    def collect_point(event, x, y, flags, param):
        nonlocal point, point_selected
        if not point_selected and event == cv2.EVENT_LBUTTONDOWN:
            point = [x, y]
            point_selected = True

    WIN_NAME = "Realsense camera"
    cv2.namedWindow(WIN_NAME)
    cv2.setMouseCallback(WIN_NAME, collect_point)
    prev_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if not point_selected:
                cv2.putText(color_image, "Click to select cluster", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(WIN_NAME, color_image)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
                    break
            else:
                if not if_init:
                    if_init = True
                    predictor.load_first_frame(color_image)

                    ann_frame_idx = 0
                    ann_obj_id = (1,)
                    labels = np.array([1], dtype=np.int32)
                    points = np.array([point], dtype=np.float32)

                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
                    )
                else:
                    out_obj_ids, out_mask_logits = predictor.track(color_image)

                #torch.cuda.synchronize() #might skew time measurements without this

                all_mask = np.zeros_like(color_image, dtype=np.uint8)
                
                out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                colored_mask = np.zeros_like(color_image, dtype=np.uint8)

                colored_mask[:, :, 0] = out_mask[:, :, 0] * 255
                colored_mask[:, :, 1] = out_mask[:, :, 0] * 0
                colored_mask[:, :, 2] = out_mask[:, :, 0] * 0

                all_mask = cv2.addWeighted(all_mask, 0, colored_mask, 1, 0)
                color_image = cv2.addWeighted(color_image, 1, all_mask, 1, 0)

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                current_time = time.time()
                frame_ms = (current_time - prev_time)
                fps = 1.0 / frame_ms
                prev_time = current_time

                cv2.putText(images, f"fps: {fps:.2f} - ms: {frame_ms * 1000:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(WIN_NAME, images)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
                    break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def main():
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
        print(f'Requested image size {args.imaage_size} is not a multple of 32 falling back to SAM2.1 default 1024')
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
   