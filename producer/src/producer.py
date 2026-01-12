from broadcaster_wrapper.broadcasting import *
import hope_server
import torch
import os
import argparse
import pyrealsense2 as rs
import producer_cli as producer_cli
import torch.multiprocessing as mp
import websocket_server

from pathlib import Path
import threading
# Set up the server

from cuda_quantizer import CudaQuantizer
from draco_wrapper import draco_bindings as dcb # temporary, for logging quality sims

from draco_wrapper.draco_wrapper import (
    DracoWrapper,
    EncodingMode,
    VisualizationMode
)

import numpy as np
import urllib.request
    
def main():
    if False:
        points = np.random.rand(150000, 3).astype(np.float32)
        colors = (np.random.rand(150000, 3) * 255).astype(np.int8)
        
        quantizer = CudaQuantizer()
        draco_roi_encoding = DracoWrapper()
        
        draco_roi_encoding.position_quantization_bits = 10
        draco_roi_encoding.color_quantization_bits    = 8
        draco_roi_encoding.speed_encode               = 10
        draco_roi_encoding.speed_decode               = 10
        
        draco_res = draco_roi_encoding.encode(points, colors, False)
        cuda_res = quantizer.encode(points, colors, (10, 10, 10), (8, 8, 8))
        
        print(f'draco:{len(draco_res) / 1024} - cuda:{len(cuda_res) / 1024}')
        
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")

    args = producer_cli.producer_cli.parse_args()

    with urllib.request.urlopen('https://ident.me') as response:
        public_ip = response.read().decode('utf8')

    server = websocket_server.SecureWebSocketServer(
        host='195.251.252.45',
        port=9003,
        cert_folder=producer_cli.CERTIFICAT_PATH,
        cert_file="server.crt",
        key_file="server.key"
    )
    
    server.start()
    
    web_server = websocket_server.SecureHTTPThreadedServer(
        host='195.251.252.45',
        port=9003 + 1,
        cert_folder=producer_cli.CERTIFICAT_PATH,
        cert_file="server.crt",
        key_file="server.key",
        directory="./../../Client/dist" 
    )
    web_server.start()


    hope_server.launch_processes(server, args, DEVICE)
        
def realsense_config() :
    # Create a context object. This object owns the handles to all connected RealSense devices
    context = rs.context()

    # Check if there are any connected devices
    if len(context.devices) == 0:
        print("No RealSense device detected.")
        exit(0)

    # List all connected devices
    for device in context.devices:
        print(f"Found device: {device.get_info(rs.camera_info.name)}")

        # Create a pipeline and config to get stream profiles
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))

        # Start pipeline to get active profiles
        profile = pipeline.start(config)

        # Query stream profiles
        sensors = device.query_sensors()
        for sensor in sensors:
            print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")
            for stream_profile in sensor.get_stream_profiles():
                if stream_profile.stream_type() == rs.stream.color or stream_profile.stream_type() == rs.stream.depth:
                    video_profile = stream_profile.as_video_stream_profile()
                    print(f"\tStream Type: {stream_profile.stream_type()}")
                    print(f"\t\tResolution: {video_profile.width()}x{video_profile.height()}")
                    print(f"\t\tFormat: {stream_profile.format()}")
                    print(f"\t\tFPS: {stream_profile.fps()}")

        pipeline.stop()

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Compiled CUDA version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    #realsense_config()
    main()
    