
from encoding import *
import torch
import os
import argparse

import producer_cli as producer_cli

from pathlib import Path
# Set up the server

def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    args = producer_cli.producer_cli.parse_args()

    enum = producer_cli.map_to_enum[args.sam2_checkpoint]
    link = producer_cli.map_to_config[enum]
    path_to_yaml = os.path.join(producer_cli.CONFIG_PATH, link[0])
    print(f"Received path_to_yaml: {path_to_yaml}")
    path_to_chkp = os.path.join(producer_cli.CHECKPOINT_PATH, Path(link[1]).name)

    if args.sam2_image_size % 32 != 0:
        print(f'Requested image size {args.sam2_image_size} is not a multple of 32 falling back to SAM2.1 default 1024')
        args.sam2_image_size = 1024

    if not os.path.exists(producer_cli.CONFIG_PATH):
        print('Config path for sam2.1 does not exist, exiting...')
        return
    
    if not os.path.exists(path_to_yaml):
        print(f'Config {link[0]} for sam2.1 does not exist, You need to download them from https://github.com/facebookresearch/sam2/tree/main/sam2/configs/sam2.1, exiting...')
        return
    
    if not os.path.exists(path_to_chkp):
        print(f'Checkpoint {path_to_chkp} is missing, downloading...')
        os.makedirs(producer_cli.CHECKPOINT_PATH, exist_ok=True)
        producer_cli.getRequest(producer_cli.CHECKPOINT_PATH, link[1])

    server = setup_server()
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    with torch.autocast(device_type=DEVICE.__str__(), dtype=torch.bfloat16):
        encode_point_cloud(server,
            args.realsense_clr_capture_width,
            args.realsense_clr_capture_height,
            args.realsense_depth_capture_width,
            args.realsense_depth_capture_height,
            args.realsense_target_fps,
            path_to_yaml=path_to_yaml, 
            path_to_chkp=path_to_chkp, 
            device=DEVICE,
            image_size=args.sam2_image_size)

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Compiled CUDA version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    main()