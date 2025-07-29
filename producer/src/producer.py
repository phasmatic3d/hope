
from encoding import *
import hope_server
import torch
import os
import argparse

import producer_cli as producer_cli
import torch.multiprocessing as mp

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
    
    server = setup_server()
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if False:
        encode_point_cloud(
            server,
            args.realsense_clr_capture_width,
            args.realsense_clr_capture_height,
            args.realsense_depth_capture_width,
            args.realsense_depth_capture_height,
            args.realsense_target_fps)
    else:
        with torch.autocast(device_type=DEVICE.__str__(), dtype=torch.bfloat16):
            hope_server.launch_processes(server, args, DEVICE)
        

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Compiled CUDA version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    main()