import argparse
import time
import threading

import cv2 
import pyrealsense2 as rs
import numpy as np

from statslogger import (
    PipelineTiming,
    CompressionStats,
    calculate_overall_time,
    make_total_time_table
)

from concurrent.futures import (
    ProcessPoolExecutor, 
    ThreadPoolExecutor,
    as_completed
)

from rich.live import Live
from rich.console import (
    Console,
    Group,
)

from mediapipe.tasks.python.vision import (
    RunningMode,
)

from gesture_recognition import (
    PointingGestureRecognizer,
    NormalizedBoundingBox,
    PixelBoundingBox
)

from .draco_wrapper import (
    DracoWrapper,
    EncodingMode,
    VizualizationMode
)

from broadcaster_wrapper import (
    setup_server,
    broadcaster
)

console = Console()

def list_realsense_modes():
    ctx = rs.context()
    if ctx.query_devices().size() == 0:
        print("No RealSense devices found.")
        return

    for dev in ctx.query_devices():
        print(f"\n=== Device: {dev.get_info(rs.camera_info.name)} "
              f"({dev.get_info(rs.camera_info.serial_number)}) ===")

        for sensor in dev.query_sensors():
            print(f"  • Sensor: {sensor.get_info(rs.camera_info.name)}")

            # Each stream-profile is a (stream-type, width×height @ fps, format) combo
            for p in sensor.get_stream_profiles():
                if not p.is_video_stream_profile():
                    continue                         # skip motion/pose streams

                vsp = p.as_video_stream_profile()
                stream_type = vsp.stream_type()
                w, h       = vsp.width(), vsp.height()
                fps        = vsp.fps()
                fmt        = vsp.format()
                print(f"     - {stream_type.name:8} "
                      f"{w:4}×{h:<4} @ {fps:>2} fps  ({fmt.name})")

def encode_point_cloud(
    server: broadcaster.ProducerServer,
    camera_rgb_width: int,
    camera_rgb_height: int,
    camera_depth_width: int,
    camera_depth_height: int,
    camera_depth_fps: int,
    camera_rgb_fps: int,
    path_to_mediapipe_task: str,
    path_to_yaml: str,
    path_to_chkp: str,
    device: str,
    image_size: int,
    visualization_mode: VizualizationMode,
    encoding_mode: EncodingMode
):

    list_realsense_modes() 
    # Setup RealSense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, camera_depth_width, camera_depth_height, rs.format.z16, camera_depth_fps)
    cfg.enable_stream(rs.stream.color, camera_rgb_width, camera_rgb_height, rs.format.rgb8, camera_rgb_fps)
    pipeline.start(cfg)

    profile = pipeline.get_active_profile()

    video_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = video_profile.get_intrinsics()

    align_to = rs.stream.color
    align = rs.align(align_to)

    prev_time = time.perf_counter()
    frame_count = 0
    fps = 0.0

    win_name = "RealSense Color"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    compression_roi_stats  = CompressionStats()
    compression_out_stats  = CompressionStats()
    compression_full_stats = CompressionStats()
    pipeline_stats         = PipelineTiming()

    process_executor = ProcessPoolExecutor(max_workers=2)
    thread_executor  = ThreadPoolExecutor(max_workers=2)

    server.listen()
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    print("Started running server thread")

    min_dist = 0.1
    max_dist = 2.0
    depth_thresh = rs.threshold_filter(min_dist, max_dist)

    # Settings
    draco_full_encoding = DracoWrapper(
        compression_stats=compression_full_stats,
    )

    draco_roi_encoding = DracoWrapper(
        compression_stats=compression_roi_stats
    )

    draco_outside_roi_encoding = DracoWrapper(
        compression_stats=compression_out_stats
    )

    gesture_recognizer = PointingGestureRecognizer(
        model_asset_path=path_to_mediapipe_task, 
        num_hands=1, 
        running_mode=RunningMode.LIVE_STREAM, 
        image_width=color_intrinsics.width,
        image_height=color_intrinsics.height,
        focal_length_x= color_intrinsics.fx,
        focal_length_y=color_intrinsics.fy,
        box_size=0.02, 
        delay_frames=10
    )



    if_sam_init = False

    # --- SUBSAMPLING LAYERS SETUP ---
    # layer 0 = 60%, layer 1 = 15%, layer 2 = 25%
    sampling_layers = [0.60, 0.15, 0.25]
    active_layers   = [True,  True,  True]

    # --- Create and configure a spatial filter --- 
    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 2)    # default=2, try 2–5
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 1)  # default=0.5
    spatial_filter.set_option(rs.option.filter_smooth_delta, 20)   # default=20
    spatial_filter.set_option(rs.option.holes_fill, 3)           # 0=none,1=small,2=medium,3=large

    
    with Live(refresh_per_second=1, screen=False) as live:
        try:
            while True:

                # ───────────────────────────────────────────────────────────────────────
                # Propagate any user‐tweaked “full” encoder settings into the ROI encoder
                # and the outside‐ROI encoder on every frame. Since we allow live key
                # presses to adjust position_quantization, color_quantization, speed_encode, etc., we must
                # copy draco_full_encoding’s current values each iteration. Otherwise
                # the ROI/non-ROI compressors would never pick up those changes.
                #
                # E.g. if you press “=” to bump position_quantization up, this ensures the next
                # frame’s ROI encoder uses that new value.
                # ───────────────────────────────────────────────────────────────────────
                #draco_roi_encoding.position_quantization_bits = draco_full_encoding.position_quantization_bits
                #draco_roi_encoding.color_quantization_bits    = draco_full_encoding.color_quantization_bits
                #draco_roi_encoding.speed_encode               = draco_full_encoding.speed_encode
                #draco_roi_encoding.speed_decode               = draco_full_encoding.speed_decode
                # ───────────────────────────────────────────────────────────────────────
                # Derive the “outside ROI” settings from the ROI encoder’s (or full’s)
                # settings. We halve the position quantization here to trade off more
                # accuracy inside the ROI vs. elsewhere. Must also run each frame so
                # any full→ROI changes flow through to this encoder as well.
                # ───────────────────────────────────────────────────────────────────────
                #draco_outside_roi_encoding.position_quantization_bits = draco_roi_encoding.position_quantization_bits
                #draco_outside_roi_encoding.color_quantization_bits    = draco_roi_encoding.color_quantization_bits
                #draco_outside_roi_encoding.speed_encode               = draco_roi_encoding.speed_encode
                #draco_outside_roi_encoding.speed_decode               = draco_roi_encoding.speed_decode

                
                frames = pipeline.wait_for_frames()

                pipeline_stats.frame_alignment_ms = time.perf_counter()
                # Align the depth frame to the color frame’s coordinate space?
                frames = align.process(frames)
                pipeline_stats.frame_alignment_ms = (time.perf_counter() - pipeline_stats.frame_alignment_ms) * 1000

                depth_frame = frames.get_depth_frame()  
                color_frame = frames.get_color_frame()    
                          
                depth_height, depth_width = depth_frame.get_height(), depth_frame.get_width()

                pipeline_stats.depth_culling_ms = time.perf_counter()
                # Apply the depth-threshold filter to drop pixels too near or too far
                depth_frame = depth_thresh.process(depth_frame)
                pipeline_stats.depth_culling_ms = (time.perf_counter() - pipeline_stats.depth_culling_ms) * 1000

                #Spatially smooth the depth map (remove noise, fill small holes)
                pipeline_stats.spatial_filter_ms = time.perf_counter()
                #depth_frame = spatial_filter.process(depth_frame) TODO: CHECK IF WORTH
                pipeline_stats.spatial_filter_ms = (time.perf_counter() - pipeline_stats.spatial_filter_ms) * 1000

                # Prepare image for display
                # Extract raw data buffers from the RealSense frames into NumPy arrays:
                #  - color_frame.get_data() gives you an H×W×3 array of RGB pixels (uint8).
                #  - depth_frame.get_data() gives you an H×W array of 16-bit depth values.
                color_img = np.asanyarray(color_frame.get_data())        
                depth_img = np.asanyarray(depth_frame.get_data())
                # OpenCV expects images in BGR channel order, but the RealSense color stream
                # hands you RGB data. So we swap R↔B here so that cv2.imshow() and all the
                # downstream OpenCV routines display and process the correct colors.
                color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR) # ?
                

                # pick which to show
                if visualization_mode is VizualizationMode.COLOR:
                    display = color_bgr
                else:
                    depth_8u = cv2.convertScaleAbs(depth_img, alpha=255.0 / depth_img.max())
                    depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                    display = depth_colormap

                # Generate point cloud
                point_cloud_time_start = time.perf_counter()
                # Create a point-cloud processor object. This handles generating 3D point data from depth frames.
                rpc = rs.pointcloud()
                # Associate the upcoming depth-to-3D mapping with the given color frame.
                # This ensures each 3D point will be textured with the correct RGB value.
                rpc.map_to(color_frame)
                # Compute the point cloud from the latest depth frame.
                # The output 'points' contains 3D coordinates (X, Y, Z) for each pixel,
                # along with indices into 'color_frame' so that each point can be colored.
                points = rpc.calculate(depth_frame) # 6 ms
                pipeline_stats.point_cloud_creation_ms = (time.perf_counter() - point_cloud_time_start) * 1000
                
                data_preparation_time_start = time.perf_counter()
                count = points.size()
                # [x0, y0, z0,  x1, y1, z1,  x2, y2, z2, …]
                vertex_buffer  = points.get_vertices()
                # [u0, v0,  u1, v1,  u2, v2, …]
                texture_buffer = points.get_texture_coordinates()
                
                # View the raw vertex buffer as a (count × 3) float32 array: each row is one 3D point (x, y, z)
                numpy_vertices = np.frombuffer(vertex_buffer,  dtype=np.float32).reshape(count, 3)
                # View the raw texture buffer as a (count × 2) float32 array: each row is one UV pair (u, v)
                numpy_textures = np.frombuffer(texture_buffer, dtype=np.float32).reshape(count, 2) # 1ms
                
                texture_scaling_time_start = time.perf_counter()
                # Precompute the scale factors once
                # Flatten and mask arrays
                # Each u,v is normalized in [0.0, 1.0]. Scale u by (width-1) to get a column index in [0, width-1],
                # then cast to int so we can index into the 2D image array
                column_coordinates = (numpy_textures[:, 0] * (depth_width - 1)).astype(np.int32)
                # Similarly, scale v by (height-1) to get a row index, then cast to int
                row_coordinates    = (numpy_textures[:, 1] * (depth_height - 1)).astype(np.int32) # 3 ms

                pixel_indices = row_coordinates * depth_width + column_coordinates

                pipeline_stats.texture_scaling_ms = (time.perf_counter() - texture_scaling_time_start) * 1000
                pipeline_stats.color_lookup_ms = time.perf_counter()
                # Flatten the H×W×3 BGR image into a (H*W)×3 array so each row is one pixel’s BGR triplet
                flat_color_pixels = color_img.reshape(-1, 3)
                colors = flat_color_pixels[pixel_indices]  # shape: (N, 3) in [R, G, B] order
                pipeline_stats.color_lookup_ms = ((time.perf_counter() - pipeline_stats.color_lookup_ms) * 1000)

                build_valid_points_start_time = time.perf_counter()
                # find valid points
                # Turn the H×W depth image into a 1-D array of length H*W
                depth_flat = depth_img.ravel()
                #depth_at_points = depth_img[row_coordinates_indices, column_coordinates_indices]

                # Build a (H*W,) boolean mask where True means:
                #  1) the depth sensor saw something (depth_flat > 0),
                #  2) the reconstructed Z coordinate is a finite number,
                #  3) and that Z coordinate is positive (in front of the camera).
                valid = (
                    (depth_flat > 0)                         # non-zero depth reading
                    & np.isfinite(numpy_vertices[:, 2])      # Z isn’t NaN or ±Inf
                    & (numpy_vertices[:, 2] > 0)             # Z > 0 (point lies in front) #TODO: Not Needed
                )
                #valid = (
                #    (depth_at_points > 0)               # depth sensor saw something
                #    & np.isfinite(numpy_vertices[:, 2]) # Z isn’t NaN or ±Inf
                #    & (numpy_vertices[:, 2] > 0)        # Z > 0 (in front of camera)
                #)
                pipeline_stats.build_valid_points_ms = (time.perf_counter() - build_valid_points_start_time) * 1000

                subsampling_time_start = time.perf_counter()
                effective_ratio = sum(r for r, on in zip(sampling_layers, active_layers) if on)

                # roll one random array and reject ~ (1‑effective_ratio) of valid points
                rnd = np.random.rand(valid.shape[0])
                subsample_mask = rnd < effective_ratio

                # Add the subsampling mask to the valid points
                valid &= subsample_mask # TODO: Maybe do not subsample the entire point cloud (maybe this has use in IMPORTANCE mode)
                pipeline_stats.subsampling_ms = (time.perf_counter() - subsampling_time_start ) * 1000


                if encoding_mode is EncodingMode.FULL:
                    # Masking
                    compression_full_stats.masking_ms = time.perf_counter()
                    points_full_frame = numpy_vertices[valid]
                    colors_full_frame = colors[valid] # 8ms
                    compression_full_stats.masking_ms = (time.perf_counter() - compression_full_stats.masking_ms ) * 1000

                    # Encode entire valid cloud
                    pipeline_stats.data_preparation_ms = (time.perf_counter() - data_preparation_time_start) * 1000 #prep end

                    if(points_full_frame.any()):
                        buffer_full = draco_full_encoding.encode(points_full_frame, colors_full_frame)

                        # Broadcast
                        # prefix with single byte to understand that we are sending one buffer
                        if len(buffer_full) != 0: 
                            server.broadcast(bytes([1]) + buffer_full)
                       
                    
                    # Logging
                    table_pipeline_stats = pipeline_stats.make_table(
                        section="End to End Pipeline Stats", 
                        show_headers=True
                    )
                    
                    table_full_compression_stats = draco_full_encoding.compression_stats.make_table(
                        section="Full Frame Compression Stats", 
                        show_headers=True
                    )

                    table_total_time = make_total_time_table(
                        total_time=calculate_overall_time(
                            pipeline_stats=pipeline_stats, 
                            compression_stats=draco_full_encoding.compression_stats
                        )
                    )

                    live.update(
                        Group(
                            table_pipeline_stats, 
                            table_full_compression_stats, 
                            table_total_time
                        )
                    )
                else:
                    buffer_roi = None
                    buffer_out = None

                    # Note currently we assume a single hand
                    gesture_recognition_start_time = time.perf_counter()
                    gesture_recognizer.recognize(color_img, int(time.time() * 1000))

                    x0 = 0
                    y0 = 0
                    x1 = 0
                    y1 = 0 

                    pixel_space_bounding_box = None
                    out_mask_logits = None

                    for bounding_box_normalized in gesture_recognizer.latest_bounding_boxes:
                        if bounding_box_normalized:
                            pixel_space_bounding_box: PixelBoundingBox = bounding_box_normalized.to_pixel(color_img.shape[1], color_img.shape[0])
                            x0 = pixel_space_bounding_box.x1
                            x1 = pixel_space_bounding_box.x2
                            y0 = pixel_space_bounding_box.y1
                            y1 = pixel_space_bounding_box.y2

                    pipeline_stats.gesture_recognition_ms = (time.perf_counter() - gesture_recognition_start_time) * 1000
 
                    cv2.rectangle(display, (x0, y0), (x1, y1), (0,255,0), 2)

                    # Importance: bin ROI vs outside
                    # Compute 2D image coordinates (row `yy`, column `xx`) for each point index:
                    # np.arange(count) gives indices [0, 1, …, count-1];
                    # divmod by depth_width turns each index into (row, col).
                    #yy, xx = np.divmod(np.arange(count), depth_width) # 6 ms
                    # Build a mask for points inside the ROI rectangle:
                    # - `valid` ensures we only consider points with real depth.
                    # - (xx>=x0) & (xx<x1) limits columns to [x0, x1).
                    # - (yy>=y0) & (yy<y1) limits rows    to [y0, y1).
                    #in_roi = (
                    #    valid
                    #    & (xx >= x0) & (xx < x1)
                    #    & (yy >= y0) & (yy < y1)
                    #)
                    compression_roi_stats.masking_ms = time.perf_counter()
                    in_roi = (
                        valid
                        & (column_coordinates >= x0) & (column_coordinates <  x1)
                        & (row_coordinates    >= y0) & (row_coordinates    <  y1)
                    )
                    # Slice out the 3D points and colors for the ROI:
                    points_in_roi  = numpy_vertices[in_roi]
                    colors_in_roi  = colors        [in_roi]
                    compression_roi_stats.masking_ms = (time.perf_counter() - compression_roi_stats.masking_ms) * 1000

                    compression_out_stats.masking_ms = time.perf_counter()
                    # Points that are valid but not in the ROI:
                    out_roi = valid & ~in_roi & subsample_mask #<--- this will help us compress non-important points
                    # Slice out the 3D points and colors for outside the ROI:
                    points_out_roi = numpy_vertices[out_roi]
                    colors_out_roi = colors        [out_roi]  
                    compression_out_stats.masking_ms = (time.perf_counter() - compression_out_stats.masking_ms) * 1000

                    pipeline_stats.data_preparation_ms = (time.perf_counter() - data_preparation_time_start) * 1000

                    # MULTIPROCESSING IMPORTANCE
                    multiprocessing_compression_time_start = time.perf_counter()
                    futures: dict[str, concurrent.futures.Future] = {}

                    if points_in_roi.size:
                        futures["roi"] = thread_executor.submit(
                            draco_roi_encoding.encode,
                            points_in_roi,
                            colors_in_roi,
                        )
                    
                    if points_out_roi.size:
                        futures["out_roi"] = thread_executor.submit(
                            draco_outside_roi_encoding.encode,
                            points_out_roi,
                            colors_out_roi,
                        )

                    for label, future in futures.items():
                        if label == "roi":
                            buffer_roi = future.result()  
                        if label == "out_roi":
                            buffer_out = future.result() 

                    pipeline_stats.multiprocessing_compression_ms = (time.perf_counter() - multiprocessing_compression_time_start) * 1000

                    # Broadcast
                    bufs = []
                    if buffer_roi:
                        bufs.append(buffer_roi)
                    if buffer_out:
                        bufs.append(buffer_out)
                    count = len(bufs)
                    for buf in bufs:
                        server.broadcast(bytes([count]) + buf) # Prefix with byte that tells us the length


                    table_pipeline_stats = pipeline_stats.make_table(
                        section="End to End Pipeline Stats", 
                        show_headers=True
                    )

                    table_roi_compression_stats = draco_roi_encoding.compression_stats.make_table(
                        section="Draco ROI Compression Stats", 
                        show_headers=True
                    )

                    table_out_compression_stats = draco_outside_roi_encoding.compression_stats.make_table(
                        section="Draco Outside ROI Compression Stats", 
                        show_headers=True
                    )

                    table_total_time = make_total_time_table(
                        total_time=calculate_overall_time(
                            pipeline_stats=pipeline_stats, 
                            compression_stats=[
                                draco_roi_encoding.compression_stats, 
                                draco_outside_roi_encoding.compression_stats
                            ]
                        )
                    )

                    live.update(
                        Group(
                            table_pipeline_stats, 
                            table_roi_compression_stats, 
                            table_out_compression_stats,
                            table_total_time
                        )
                    )
                
            
                frame_count += 1
                now = time.perf_counter()
                if (frame_count >= 30):
                    fps = frame_count // (now - prev_time)
                    prev_time = now
                    frame_count = 0

                # draw FPS
                cv2.putText(display, f"FPS: {fps}", (camera_rgb_width -200,camera_rgb_height -30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # draw compression stats
                text = str(draco_full_encoding.compression_stats)
                x, y = 10, 20
                line_height = 20
                for i, line in enumerate(text.splitlines()):
                    cv2.putText(
                        display,
                        line,
                        (x, y + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                # now draw layer_str just below the last stats line
                layer_str = " ".join(
                    f"L{i}{' ON' if active_layers[i] else ' OFF'}({int(sampling_layers[i]*100)}%)"
                    for i in range(len(sampling_layers))
                )
                # compute start_y: one line below the last stats line
                n_lines = len(text.splitlines())
                start_y = y + n_lines * line_height
                cv2.putText(
                    display,
                    layer_str,
                    (x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                cv2.imshow(win_name, display)      
                key = cv2.waitKey(1) # this takes 20 ms
                if key in (ord('q'), 27):  # q or Esc
                    break
                ## Adjust settings
                #elif key == ord('f'):
                #    # Toggle mode (IMPORTANCE)
                #    encoding_mode = EncodingMode.FULL if encoding_mode is EncodingMode.IMPORTANCE else EncodingMode.IMPORTANCE
                #elif key == ord('d'):
                #    #Toggle mode (Visualization: Depth vs Color)
                #    visualization_mode = (
                #        visualization_mode.DEPTH
                #        if visualization_mode is visualization_mode.COLOR
                #        else visualization_mode.COLOR
                #    )
                #elif key == ord('='):
                #    draco_full_encoding.posQuant = min(draco_full_encoding.posQuant+1, 20)
                #elif key == ord('-'):
                #    draco_full_encoding.posQuant = max(draco_full_encoding.posQuant-1, 1)
                #elif key == ord(']'):
                #    draco_full_encoding.colorQuant = min(draco_full_encoding.colorQuant+1, 16)
                #elif key == ord('['):
                #    draco_full_encoding.colorQuant = max(draco_full_encoding.colorQuant-1, 1)
                #elif key == ord('.'):
                #    draco_full_encoding.speedEncode = min(draco_full_encoding.speedEncode+1, 10)
                #    draco_full_encoding.speedDecode = draco_full_encoding.speedEncode
                #elif key == ord(','):
                #    draco_full_encoding.speedEncode = max(draco_full_encoding.speedEncode-1, 0)
                #    draco_full_encoding.speedDecode = draco_full_encoding.speedEncode
                #elif key == ord(' '):
                #    # save full point cloud PLY
                #    points.export_to_ply("snapshot.ply", color_frame)
                #    print("Saved full point cloud to snapshot.ply")
                # LAYERS TOGGLE
                #elif key == ord('1'):
                #    active_layers[0] = not active_layers[0]
                #elif key == ord('2'):
                #    active_layers[1] = not active_layers[1]
                #elif key == ord('3'):
                #    active_layers[2] = not active_layers[2]
        finally:
            pipeline.stop()
            # cleanly shut down the WebSocket server
            server.stop()          # calls stop_listening() + stop() in C++
            server_thread.join()   # wait for run() to return

def main():

    parser = argparse.ArgumentParser(
        description="Real-time RGB-D capture, Draco-compress & broadcast"
    )

    parser.add_argument(
        "--camera-rgb-width",
        type=int,
        default=640,
        help="Color stream width (pixels)",
    )

    parser.add_argument(
        "--camera-rgb-height",
        type=int,
        default=480,
        help="Color stream height (pixels)",
    )

    parser.add_argument(
        "--camera-depth-width",
        type=int,
        default=848,
        help="Depth stream width (pixels)",
    )

    parser.add_argument(
        "--camera-depth-height",
        type=int,
        default=480,
        help="Depth stream height (pixels)",
    )
    parser.add_argument(
        "--camera-depth-fps",
        type=int,
        default=10,
        help="Depth stream framerate (fps)",
    )
    parser.add_argument(
        "--camera-rgb-fps",
        type=int,
        default=30,
        help="Color stream framerate (fps)",
    )

    parser.add_argument(
        "--config-yaml",
        type=str,
        default="./configs",
        help="Path to your YAML config file",
    )

    parser.add_argument(
        "--mediapipe-task-path",
        type=str,
        default="./draco_wrapper/hand_landmarker.task",
        help="Path to the mediapipe task.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./draco_wrapper/checkpoints",
        help="Path to your model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (e.g. 'cpu' or 'cuda:0')",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize dimension for model input",
    )

    parser.add_argument(
        "--server-host",        
        type=str, 
        default="https://localhost",
        help="Broadcast server host"
    )

    parser.add_argument(
        "--server-port",        
        type=int, 
        default=5555,
        help="Broadcast server port"
    )

    parser.add_argument(
        "--server-write-to-csv",        
        type=int, 
        choices=[0, 1],
        default=0,
        help="Write latency values to csv"
    )

    parser.add_argument(
        "--server-use-pings-for-rtt",        
        type=int, 
        choices=[0, 1],
        default=1,
        help="Calculate RTT using pings instead of timestamps."
    )

    parser.add_argument(
        "--visualization-mode",
        choices=["color", "depth"],
        default="color",
        help="Which image to show: color or depth",
    )

    parser.add_argument(
        "--encoding-mode",
        choices=["full", "importance"],
        default="full",
        help="Encoding mode for the encoder.",
    )

    args = parser.parse_args()

    server = setup_server(
        url=args.server_host, 
        port=args.server_port, 
        write_to_csv=bool(args.server_write_to_csv), 
        use_pings_for_rtt=bool(args.server_use_pings_for_rtt)
    )

    if args.visualization_mode == "color":
        visualization_mode = VizualizationMode.COLOR
    elif args.visualization_mode == "depth":
        visualization_mode = VizualizationMode.DEPTH

    if args.encoding_mode == "full":
        encoding_mode = EncodingMode.FULL
    elif args.encoding_mode == "importance":
        encoding_mode = EncodingMode.IMPORTANCE

    encode_point_cloud(
        server=server,
        camera_rgb_width=args.camera_rgb_width,
        camera_rgb_height=args.camera_rgb_height,
        camera_depth_width=args.camera_depth_width,
        camera_depth_height=args.camera_depth_height,
        camera_depth_fps=args.camera_depth_fps,
        camera_rgb_fps=args.camera_rgb_fps,
        path_to_mediapipe_task=args.mediapipe_task_path,
        path_to_yaml=args.config_yaml,
        path_to_chkp=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        visualization_mode=visualization_mode,
        encoding_mode=encoding_mode
    )


if __name__ == "__main__":
    main()