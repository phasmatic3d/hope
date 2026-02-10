import hope_server
import torch
import websocket_server
import pyrealsense2 as rs
import producer_cli as producer_cli
import offline_compression
import time

from pathlib import Path


def wait_for_first_client(server) -> None:
    """Block offline processing until at least one websocket client is connected."""
    # Offline streaming starts only after the first client is present.
    while server.get_client_count() == 0:
        print("[offline] waiting for first websocket client on wss://<host>:9003 ...")
        time.sleep(0.5)


def main():  
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")

    args = producer_cli.producer_cli.parse_args()
    args.device = DEVICE

    if args.offline_mode:
        # Offline mode serves the web client and streams compressed chunks over WSS.
        websocket_server.generate_self_signed_cert(Path("."))
        offline_server = websocket_server.SecureWebSocketServer(
            host=websocket_server.get_local_ip(),
            port=9003,
            cert_folder=producer_cli.CERTIFICAT_PATH,
            cert_file="server.crt",
            key_file="server.key",
        )
        offline_web_server = websocket_server.SecureHTTPThreadedServer(
            host=websocket_server.get_local_ip(),
            port=9003 + 1,
            cert_folder=producer_cli.CERTIFICAT_PATH,
            cert_file="server.crt",
            key_file="server.key",
            directory="./../Client/dist",
        )
        offline_server.start()
        offline_web_server.start()
        try:
            # Frame 0 waits until at least one websocket client is connected.
            wait_for_first_client(offline_server)
            offline_compression.run_offline_compression(args, offline_server)
        finally:
            # Stop both services after offline compression exits.
            offline_web_server.stop()
            offline_server.stop()
        return
    websocket_server.generate_self_signed_cert(Path("."))
    webrtc = False
    server_cls = websocket_server.SecureWebRTCServer if webrtc else websocket_server.SecureWebSocketServer
      
    server = server_cls(
        host=websocket_server.get_local_ip(),
        port=9003,
        cert_folder=producer_cli.CERTIFICAT_PATH,
        cert_file="server.crt",
        key_file="server.key")
    
    server.start()
    
    web_server = websocket_server.SecureHTTPThreadedServer(
        host=websocket_server.get_local_ip(),
        port=9003 + 1,
        cert_folder=producer_cli.CERTIFICAT_PATH,
        cert_file="server.crt",
        key_file="server.key",
        directory="./../Client/dist" )
    
    web_server.start()
    hope_server.launch_processes(server, args, DEVICE)
        
def realsense_config() :
    context = rs.context()

    if len(context.devices) == 0:
        print("No RealSense device detected.")
        exit(0)

    for device in context.devices:
        print(f"Found device: {device.get_info(rs.camera_info.name)}")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))

        profile = pipeline.start(config)

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


    