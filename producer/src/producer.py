from broadcaster_wrapper.broadcasting import *
import hope_server
import torch
import os
import argparse
import pyrealsense2 as rs
import producer_cli as producer_cli
import torch.multiprocessing as mp

from pathlib import Path
import threading

from recording import compress_pc_set_full, compress_pc_set_importance
# Set up the server

def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    args = producer_cli.producer_cli.parse_args()

    # Toggle background removal
    background_removal_enabled = not args.disable_background_removal

    # Offline IMPORTANCE compression mode
    if args.compress_PC_set_IMPORTANCE is not None:

        set_indices = [int(idx.strip()) for idx in args.compress_PC_set_IMPORTANCE.split(',') if idx.strip()]
        export_root = Path(__file__).resolve().parent.parent / "exported_PCs"

        sam_config = None
        yolo_config = None
        if args.cluster_predictor == "sam2":
            enum = producer_cli.map_to_enum[args.sam2_checkpoint]
            path_to_yaml = Path("configs") / producer_cli.map_to_config[enum][0]
            path_to_chkp = Path("checkpoints") / Path(producer_cli.map_to_config[enum][1]).name

            sam_config = {
                "config_file": path_to_yaml.name,
                "config_path": str(path_to_yaml.parent),
                "checkpoint_path": path_to_chkp,
                "device": DEVICE,
                "image_size": args.sam2_image_size,
                "box_fraction": args.importance_box_fraction,
            }
        else:
            yolo_config = {
                "device": DEVICE,
                "size": args.yolo_size,
            }

        compress_pc_set_importance(
            set_indices=set_indices,
            export_root=export_root,
            min_depth=hope_server.MIN_DEPTH_M,
            max_depth=hope_server.MAX_DEPTH_M,
            query_point=args.importance_query_point,
            bandwidth_mbps=args.max_bandwidth_mbps,
            target_frame_rate=args.target_frame_rate,
            predictor_type=args.cluster_predictor,
            sam_config=sam_config,
            yolo_config=yolo_config,
            remove_background=background_removal_enabled,
            in_roi_pos_bits=args.importance_in_roi_pos_bits,
            out_roi_pos_bits=args.importance_out_roi_pos_bits,
            in_roi_color_bits=args.importance_in_roi_color_bits,
            out_roi_color_bits=args.importance_out_roi_color_bits,
        )
        return

    # Offline compression mode
    if args.compress_PC_set_FULL is not None:
        export_root = Path(__file__).resolve().parent.parent / "exported_PCs"
        compress_pc_set_full(
            args.compress_PC_set_FULL,
            export_root,
            hope_server.MIN_DEPTH_M,
            hope_server.MAX_DEPTH_M,
            bandwidth_mbps=args.max_bandwidth_mbps,
            target_frame_rate=args.target_frame_rate,
            subsample_frames=args.subsample_frames,
            remove_background=background_removal_enabled,
        )
        return

    server = setup_server(
        args.server_port,
        args.server_host,
        args.server_write_to_csv,
        args.server_use_pings_for_rtt
    )
    server.listen()
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
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
