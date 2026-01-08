from enum import Enum, auto
from pyexpat import model
import time
import os

import pandas as pd

import numpy as np
import producer_cli as producer_cli

import mediapipe as mediap
from mediapipe.framework.formats import landmark_pb2

from pathlib import Path
from typing import Tuple
from broadcaster_wrapper.broadcasting import *

import pyrealsense2 as rs
import cv2
import sam2_camera_predictor as sam2_camera
from ultralytics import YOLOE

from draco_wrapper.draco_wrapper import (
    DracoWrapper,
    EncodingMode,
    VizualizationMode
)

from utils import (
    write_pointcloud_ply
)

from recording import (
    RecordingManager,
    compute_effective_subsample_ratio,
    compute_importance_subsample_ratio,
    BYTES_PER_POINT,
)


from draco_wrapper import draco_bindings as dcb # temporary, for logging quality sims

from broadcaster_wrapper import (
    broadcaster
)


from concurrent.futures import (
    ProcessPoolExecutor, 
    ThreadPoolExecutor,
)

from rich.live import Live
from rich.console import Group

from statslogger import (
    PipelineTiming,
    CompressionStats,
    calculate_overall_time,
    make_total_time_table,
    write_stats_csv,
    write_simulation_csv,
    write_quality_simulation_csv,
    generate_combinations
)

import multiprocessing as mp
from multiprocessing import shared_memory

from collections import deque

from dataclasses import dataclass

from mediapipe.tasks.python.vision import (RunningMode,)

from gesture_recognition import (
    PointingGestureRecognizer,
)

import torch
import torch.nn.functional as F

DEBUG = True

MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 1.3

def camera_process(
        server: broadcaster.ProducerServer,
        shared_frame_name,
        shared_cluster_name,
        shared_roi_name,
        cmr_clr_width: int,
        cmr_clr_height: int,
        cmr_depth_width: int,
        cmr_depth_height: int,
        cmr_fps: int,
        bandwidth_mbps: float,
        target_frame_rate: float,
        subsample_frames: bool,
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event,
        ready_roi_event: mp.Event,
        simulation: bool,
        record_frame_count: int) :
    
    def apply_combo_settings(combo): # Helper function for setting a combination for simulations
        if encoding_mode == EncodingMode.FULL:
            draco_full_encoding.position_quantization_bits = combo["pos_bits"]
            draco_full_encoding.color_quantization_bits    = combo["col_bits"]
            draco_full_encoding.speed_encode               = combo["encoding_speed"]
            draco_full_encoding.speed_decode               = combo["decoding_speed"]
        elif encoding_mode == EncodingMode.IMPORTANCE:
            draco_roi_encoding.position_quantization_bits        = combo["pos_bits_in"]
            draco_roi_encoding.color_quantization_bits           = combo["col_bits_in"]
            draco_roi_encoding.speed_encode                      = combo["encoding_speed_in"]
            draco_roi_encoding.speed_decode                      = combo["decoding_speed_in"]
            draco_outside_roi_encoding.position_quantization_bits = combo["pos_bits_out"]
            draco_outside_roi_encoding.color_quantization_bits    = combo["col_bits_out"]
            draco_outside_roi_encoding.speed_encode               = combo["encoding_speed_out"]
            draco_outside_roi_encoding.speed_decode               = combo["decoding_speed_out"]
    
    global DEBUG

    # Hyperparameters to move to argparse
    visualization_mode = VizualizationMode.COLOR
    encoding_mode      = EncodingMode.FULL

    frame_stats_buffer = deque(maxlen=30) # buffer for dynamic stats logging (CSV)

    # simulation settings for simulation csv logging
    simulation_combos   = []
    performance_simulation_index    = None
    performance_simulation_buffer   = deque(maxlen=30)

    quality_simulation_index    = None
    quality_simulation_buffer   = deque(maxlen=30)

    thread_executor = ThreadPoolExecutor(max_workers=2)

    recording_manager = RecordingManager(
        export_root=Path(__file__).resolve().parent.parent / "exported_PCs",
        frame_target=record_frame_count,
    )
    
    if DEBUG:
        win_name = "RealSense vis"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        cursor_position = [0, 0]

        def handle_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                cursor_position[0] = x
                cursor_position[1] = y

        cv2.setMouseCallback(win_name, handle_mouse)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, cmr_depth_width, cmr_depth_height, rs.format.z16, cmr_fps)
    cfg.enable_stream(rs.stream.color, cmr_clr_width, cmr_clr_height, rs.format.rgb8, cmr_fps)
    pipeline.start(cfg)

    profile = pipeline.get_active_profile()

    video_profile = profile.get_stream(rs.stream.color) \
                       .as_video_stream_profile()
    
    color_intrinsics = video_profile.get_intrinsics()

    gesture_recognizer = PointingGestureRecognizer(
        model_asset_path="hand_landmarker.task", 
        num_hands=1, 
        running_mode=RunningMode.LIVE_STREAM, 
        image_width=color_intrinsics.width,
        image_height=color_intrinsics.height,
        focal_length_x= color_intrinsics.fx,
        focal_length_y=color_intrinsics.fy,
        box_size=0.02, 
        delay_frames=10)
    
    min_dist = MIN_DEPTH_M
    max_dist = MAX_DEPTH_M
    depth_thresh = rs.threshold_filter(min_dist, max_dist)

    align_to = rs.stream.color
    align = rs.align(align_to)
    rpc = rs.pointcloud()

    
    frame_id = 0
    

    
    prev_time = time.perf_counter()
    frame_count = 0
    fps = 0.0


    compression_roi_stats  = CompressionStats()
    compression_out_stats  = CompressionStats()
    compression_full_stats = CompressionStats()
    pipeline_stats = PipelineTiming()

    # Settings
    draco_full_encoding = DracoWrapper(
        compression_stats=compression_full_stats
    )

    draco_roi_encoding = DracoWrapper(
        compression_stats=compression_roi_stats
    )

    draco_outside_roi_encoding = DracoWrapper(
        compression_stats=compression_out_stats
    )

    # SAM
    sf = shared_memory.SharedMemory(name=shared_frame_name)
    sc = shared_memory.SharedMemory(name=shared_cluster_name)
    sroi = shared_memory.SharedMemory(name=shared_roi_name)

    shared_frame = np.ndarray((cmr_clr_height, cmr_clr_width, 3), dtype=np.uint8, buffer=sf.buf)
    shared_cluster = np.ndarray((cmr_clr_height, cmr_clr_width, 1), dtype=np.bool, buffer=sc.buf)
    shared_roi = np.ndarray((4,), dtype=np.int32, buffer=sroi.buf)

    ready_frame_event.clear()
    prev_cluster = shared_cluster.copy()
    roi = None


    broadcast_round = 0 # Keep track of broadcasting round to query cpp csv for logging
    batch = 0
    effective_ratio = 1.0
    
    try:
        while not stop_event.is_set():
            # Sync hyperparameter changes
            draco_roi_encoding.position_quantization_bits = draco_full_encoding.position_quantization_bits
            draco_roi_encoding.color_quantization_bits    = draco_full_encoding.color_quantization_bits
            draco_roi_encoding.speed_encode               = draco_full_encoding.speed_encode
            draco_roi_encoding.speed_decode               = draco_full_encoding.speed_decode


            frame_id += 1
            frame_count += 1
            frame_capture_start = time.perf_counter() 
            frames = pipeline.wait_for_frames()

            pipeline_stats.frame_preparation_ms = time.perf_counter()

            pipeline_stats.frame_alignment_ms = time.perf_counter()
            frames = align.process(frames)
            pipeline_stats.frame_alignment_ms = (time.perf_counter() - pipeline_stats.frame_alignment_ms) * 1000

            depth_frame = frames.get_depth_frame()  
            color_frame = frames.get_color_frame()    
                        
            depth_height, depth_width = depth_frame.get_height(), depth_frame.get_width()

            pipeline_stats.depth_culling_ms = time.perf_counter()
            # Apply the depth-threshold filter to drop pixels too near or too far
            if(encoding_mode != EncodingMode.NONE):
                depth_frame = depth_thresh.process(depth_frame)
            pipeline_stats.depth_culling_ms = (time.perf_counter() - pipeline_stats.depth_culling_ms) * 1000

            # Prepare image for display
            # Extract raw data buffers from the RealSense frames into NumPy arrays:
            #  - color_frame.get_data() gives you an H×W×3 array of RGB pixels (uint8).
            #  - depth_frame.get_data() gives you an H×W array of 16-bit depth values.
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())

            depth_flat = depth_img.ravel()

            display = color_img

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
            points = rpc.calculate(depth_frame)
            pipeline_stats.point_cloud_creation_ms = (time.perf_counter() - point_cloud_time_start) * 1000

            pipeline_stats.frame_preparation_ms = (time.perf_counter() - pipeline_stats.frame_preparation_ms) * 1000 # frame prep end
            
            
            pipeline_stats.data_preparation_ms = time.perf_counter()
            num_points = points.size()

            # [x0, y0, z0,  x1, y1, z1,  x2, y2, z2, …]
            vertex_buffer  = points.get_vertices()
            # [u0, v0,  u1, v1,  u2, v2, …]
            texture_buffer = points.get_texture_coordinates()

            # View the raw vertex buffer as a (num_points × 3) float32 array: each row is one 3D point (x, y, z)
            vertices = np.frombuffer(vertex_buffer,  dtype=np.float32).reshape(num_points, 3)
            # View the raw texture buffer as a (num_points × 2) float32 array: each row is one UV pair (u, v)
            texcoords = np.frombuffer(texture_buffer, dtype=np.float32).reshape(num_points, 2) # 1ms

            texture_scaling_time_start = time.perf_counter()
            # Precompute the scale factors once
            # Flatten and mask arrays
            # Each u,v is normalized in [0.0, 1.0]. Scale u by (width-1) to get a column index in [0, width-1],
            # then cast to int so we can index into the 2D image array
            column_coordinates = (texcoords[:, 0] * (depth_width - 1)).astype(np.int32)
            # Similarly, scale v by (height-1) to get a row index, then cast to int
            row_coordinates    = (texcoords[:, 1] * (depth_height - 1)).astype(np.int32) # 3 ms

            pixel_indices = row_coordinates * depth_width + column_coordinates

            pipeline_stats.texture_scaling_ms = (time.perf_counter() - texture_scaling_time_start) * 1000

            pipeline_stats.color_lookup_ms = time.perf_counter()
            # Flatten the H×W×3 BGR image into a (H*W)×3 array so each row is one pixel’s BGR triplet
            flat_color_pixels = color_img.reshape(-1, 3)
            colors = flat_color_pixels[pixel_indices]  # shape: (N, 3) in [R, G, B] order
            pipeline_stats.color_lookup_ms = ((time.perf_counter() - pipeline_stats.color_lookup_ms) * 1000)

            valid_points_mask = (
                (depth_flat > 0)
                & np.isfinite(vertices[:, 2])
            )

            valid = valid_points_mask

            subsampling_time_start = time.perf_counter()
            effective_ratio = 1.0
            subsample_mask = np.ones(valid.shape[0], dtype=bool)

            if encoding_mode == EncodingMode.NONE:
                valid_points_count = int(np.count_nonzero(valid))
                
                if subsample_frames:
                    effective_ratio = compute_effective_subsample_ratio(
                        valid_points_count,
                        bandwidth_mbps,
                        target_frame_rate,
                        bytes_per_point=BYTES_PER_POINT,
                    )


                    subsample_mask = np.random.rand(valid.shape[0]) < effective_ratio
                    print(f"[NONE] Effective subsample ratio applied: {effective_ratio:.4f}")

                pipeline_stats.subsampling_ms = (time.perf_counter() - subsampling_time_start ) * 1000

           
                pipeline_stats.data_preparation_ms = (time.perf_counter() - pipeline_stats.data_preparation_ms) * 1000 #prep end

                valid_mask = valid & subsample_mask
                if recording_manager.is_active:
                    recording_manager.capture_frame(
                        vertices,
                        colors,
                        color_img,
                        depth_img,
                        valid_mask,
                        capture_time_ms=(time.perf_counter() - frame_capture_start) * 1000,
                        record_native_color=True,
                    )
                raw_points = vertices[valid_mask]
                raw_cols = colors[valid_mask]

                if(raw_points.size > 0):
                    payload = bytes([0]) + raw_points.tobytes() + raw_cols.tobytes()
                    server.broadcast(payload)

                entry = server.wait_for_entry(broadcast_round)
                if entry:
                    broadcast_round += 1
                    pipeline_stats.approximate_rtt_ms = entry.approximate_rtt_ms

            else:

                build_valid_points_start_time = time.perf_counter()
                pipeline_stats.build_valid_points_ms = (time.perf_counter() - build_valid_points_start_time) * 1000

                if encoding_mode == EncodingMode.FULL:
                    valid_points_count = int(np.count_nonzero(valid))
                    effective_ratio = compute_effective_subsample_ratio(
                        valid_points_count,
                        bandwidth_mbps,
                        target_frame_rate,
                        bytes_per_point=BYTES_PER_POINT,
                    )
                    subsample_mask = np.random.rand(valid.shape[0]) < effective_ratio
                    print(f"[FULL] Effective subsample ratio applied: {effective_ratio:.4f}")

                if encoding_mode != EncodingMode.IMPORTANCE:
                    valid &= subsample_mask
                    if recording_manager.is_active:
                        recording_manager.capture_frame(
                            vertices,
                            colors,
                            color_img,
                            depth_img,
                            valid,
                            capture_time_ms=(time.perf_counter() - frame_capture_start) * 1000,
                        )

                if(encoding_mode == EncodingMode.IMPORTANCE):
                    # ---GESTURE RECOGNITION---
                    gesture_recognizer.recognize(display, frame_id)

                    if DEBUG and False:
                        def ensure_landmark_list(data):
                            if isinstance(data, landmark_pb2.NormalizedLandmarkList):
                                return data
                            elif isinstance(data, list):
                                # Rebuild
                                return landmark_pb2.NormalizedLandmarkList(
                                    landmark=[
                                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in data
                                    ]
                                )
                            else:
                                raise TypeError("Unexpected landmark data type")

                        mp_drawing = mediap.solutions.drawing_utils
                        mp_hands = mediap.solutions.hands
                        with gesture_recognizer.lock:
                            if gesture_recognizer.cb_result is not None:
                                for hand_landmark in gesture_recognizer.cb_result:
                                    hand_landmark = ensure_landmark_list(hand_landmark)
                                    mp_drawing.draw_landmarks(
                                        image=display,
                                        landmark_list=hand_landmark,
                                        connections=mp_hands.HAND_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))

                    for bounding_box_normalized in gesture_recognizer.latest_bounding_boxes:
                        if bounding_box_normalized:
                            roi: np.array = bounding_box_normalized.to_pixel(display.shape[1], display.shape[0], True)
                            shared_roi[:] = roi
                            ready_roi_event.set()

                    shared_frame[:] = display
                    ready_frame_event.set()

                    #TODO: we need to find some work todo here in order to hide latency between processes

                    if ready_cluster_event.is_set():
                        prev_cluster = shared_cluster.copy()
                        ready_cluster_event.clear()

                    if DEBUG:
                        display[prev_cluster[:, :, 0], 2] = 255

                    flat_cluster = prev_cluster.flatten()

                    compression_full_stats.masking_ms = time.perf_counter()
                    non_roi_candidates = valid & ~flat_cluster
                    roi_count = int(np.count_nonzero(valid & flat_cluster))
                    candidate_count = int(np.count_nonzero(non_roi_candidates))

                    effective_ratio = compute_importance_subsample_ratio(
                        roi_count,
                        candidate_count,
                        bandwidth_mbps,
                        target_frame_rate,
                        draco_roi_encoding.position_quantization_bits,
                        draco_roi_encoding.color_quantization_bits,
                        draco_outside_roi_encoding.position_quantization_bits,
                        draco_outside_roi_encoding.color_quantization_bits,
                    )

                    subsample_mask = np.zeros_like(valid, dtype=bool)
                    if candidate_count:
                        candidate_indices = np.flatnonzero(non_roi_candidates)
                        sampled = np.random.rand(candidate_count) < effective_ratio
                        subsample_mask[candidate_indices] = sampled

                    print(f"[IMPORTANCE] Effective subsample ratio applied: {effective_ratio:.4f}")

                    in_roi = valid & flat_cluster
                    out_roi = non_roi_candidates & subsample_mask
                    points_in_roi = vertices[in_roi]; colors_in_roi = colors[in_roi]
                    points_out_roi = vertices[out_roi]; colors_out_roi = colors[out_roi]
                    if recording_manager.is_active:
                        # Save ROI frames with only out-of-ROI subsampled.
                        recording_manager.capture_frame(
                            vertices,
                            colors,
                            color_img,
                            depth_img,
                            in_roi | out_roi,
                            capture_time_ms=(time.perf_counter() - frame_capture_start) * 1000,
                        )
                    compression_full_stats.masking_ms = (time.perf_counter() - compression_full_stats.masking_ms ) * 1000

                    pipeline_stats.data_preparation_ms = (time.perf_counter() - pipeline_stats.data_preparation_ms) * 1000 #prep end
                
                    # MULTIPROCESSING IMPORTANCE
                    multiprocessing_compression_time_start = time.perf_counter()
                    futures: dict[str, concurrent.futures.Future] = {}
                
                    buffer_roi = None
                    buffer_out = None
                    if points_in_roi.size:
                        futures["roi"] = thread_executor.submit(
                            draco_roi_encoding.encode,
                            points_in_roi,
                            colors_in_roi,
                        )
                    else:
                        buffer_roi = b""
                    
                    if points_out_roi.size:
                        futures["out_roi"] = thread_executor.submit(
                            draco_outside_roi_encoding.encode,
                            points_out_roi,
                            colors_out_roi,
                        )
                    else:
                        buffer_out = b""

                    for label, future in futures.items():
                        if label == "roi":
                            buffer_roi = future.result()  
                        if label == "out_roi":
                            buffer_out = future.result() 
                    pipeline_stats.multiprocessing_compression_ms = (time.perf_counter() - multiprocessing_compression_time_start) * 1000

                    # Broadcast
                    buffers = []
                    if buffer_roi:
                        buffers.append(buffer_roi)
                    if buffer_out:
                        buffers.append(buffer_out)
                    count = len(buffers)

                    any_broadcasted = False
                    for buffer in buffers:
                        any_broadcasted |= server.broadcast_batch(batch, bytes([count]) + buffer)
                        entry = server.wait_for_entry(broadcast_round)
                        if entry:
                            broadcast_round += 1
                            pipeline_stats.approximate_rtt_ms += entry.approximate_rtt_ms

                    if any_broadcasted:
                        batch += 1


                else: # ENCODE THE FULL FRAME
                    # Masking
                    compression_full_stats.masking_ms = time.perf_counter()
                    points_full = vertices[valid]
                    colors_full = colors[valid]
                    compression_full_stats.masking_ms = (time.perf_counter() - compression_full_stats.masking_ms ) * 1000
                

                    # Encode entire valid cloud
                    pipeline_stats.data_preparation_ms = (time.perf_counter() - pipeline_stats.data_preparation_ms) * 1000 #prep end
                    if(points_full.any()):
                        buffer_full = draco_full_encoding.encode(points_full, colors_full)
                        server.broadcast(bytes([1]) + buffer_full) # prefix with single byte to understand that we are sending one buffer
                    
                    entry = server.wait_for_entry(broadcast_round)
                    if entry:
                        broadcast_round += 1
                       
                        
            

            # Logging and display
            if DEBUG:
                if (encoding_mode == EncodingMode.IMPORTANCE):
                    points_full = vertices[out_roi & subsample_mask | in_roi ]
                    colors_full = colors[out_roi & subsample_mask | in_roi]

                    raw_size_in  = points_in_roi.nbytes  + colors_in_roi.nbytes
                    raw_size_out = points_out_roi.nbytes + colors_out_roi.nbytes
                    raw_size = raw_size_in + raw_size_out
                    encoded_size = compression_roi_stats.encoded_bytes + compression_out_stats.encoded_bytes
                elif (encoding_mode == EncodingMode.FULL):
                    raw_size = points_full.nbytes  + colors_full.nbytes

                    encoded_size = compression_full_stats.encoded_bytes

                # Logging
                if(entry): # If broadcasting
                    frame_stats_buffer.append({
                        "frame_preparation_ms":         pipeline_stats.frame_preparation_ms,
                        "data_preparation_ms":         pipeline_stats.data_preparation_ms,
                        "multiprocessing_compression_ms": pipeline_stats.multiprocessing_compression_ms,
                        "one_way_ms":                       entry.one_way_ms,
                        "decode_ms":                        entry.pure_decode_ms,
                        "geometry_upload_ms":                entry.pure_geometry_upload_ms,
                        "render_ms":                        entry.pure_render_ms,
                        "full_encode_ms":              compression_full_stats.compression_ms,
                        "roi_encode_ms":               compression_roi_stats.compression_ms,
                        "out_encode_ms":           compression_out_stats.compression_ms,
                        "num_points":                  int(num_points),
                        "full_points":                  int(compression_full_stats.number_of_points),
                        "raw_points_size":                  int(raw_size),
                        "encoded_points_size":              int (encoded_size),
                        "in_roi_points":                int(compression_roi_stats.number_of_points),
                        "out_roi_points":               int(compression_out_stats.number_of_points)
                    })

                    # Performance simulation
                    if performance_simulation_index is not None: # only if simulation is running


                        performance_simulation_buffer.append({
                            "frame_preparation_ms":         pipeline_stats.frame_preparation_ms,
                            "data_preparation_ms":          pipeline_stats.data_preparation_ms,
                            "multiprocessing_compression_ms": pipeline_stats.multiprocessing_compression_ms,
                            "one_way_ms":                       entry.one_way_ms,
                            "decode_ms":                        entry.pure_decode_ms,
                            "geometry_upload_ms":                  entry.pure_geometry_upload_ms,
                            "render_ms":                  entry.pure_render_ms,
                            "full_encode_ms":               compression_full_stats.compression_ms,
                            "roi_encode_ms":                compression_roi_stats.compression_ms,
                            "out_encode_ms":            compression_out_stats.compression_ms,
                            "num_points":                   int(num_points),
                            "full_points":                  int(compression_full_stats.number_of_points),
                            "raw_points_size":                  int(raw_size),
                            "encoded_points_size":              int (encoded_size),
                            "in_roi_points":                int(compression_roi_stats.number_of_points),
                            "out_roi_points":               int(compression_out_stats.number_of_points),
                            
                        })

                        performance_simulation_index = write_simulation_csv(
                            performance_simulation_buffer,
                            simulation_combos,
                            performance_simulation_index,
                            encoding_mode,
                            (cmr_clr_width, cmr_clr_height),
                            (cmr_depth_width, cmr_depth_height),
                        )
                        if performance_simulation_index is not None:
                            apply_combo_settings(simulation_combos[performance_simulation_index])
                        else:
                            print("SIMULATION COMPLETE ✅")


                # now append into quality buffer if running quality simulation
                if quality_simulation_index is not None :
                    # decode points
                    if encoding_mode == EncodingMode.FULL:
                        decoded_positions, decoded_colors = dcb.decode_pointcloud(buffer_full)            
                    elif encoding_mode == EncodingMode.IMPORTANCE and buffer_roi:
                        #recalculate points full here (we don't have it from earlier)

                        points_full = vertices[out_roi & subsample_mask | in_roi ]
                        colors_full = colors[out_roi & subsample_mask | in_roi]

                        decoded_positions_roi, decoded_colors_roi = dcb.decode_pointcloud(buffer_roi)
                        decoded_positions_out, decoded_colors_out = dcb.decode_pointcloud(buffer_out)
                        decoded_positions = np.vstack((decoded_positions_roi, decoded_positions_out)) 
                        decoded_colors = np.vstack((decoded_colors_roi, decoded_colors_out))  


                    position_diffs = np.linalg.norm(points_full - decoded_positions, axis=1)

                    radii = np.linalg.norm(points_full, axis=1)
                    max_radius = radii.max()
                    relative_position_diffs = position_diffs / max_radius

                    color_errs = np.abs(colors_full.astype(int) - decoded_colors.astype(int))
                    relative_color_errs = color_errs / 255.0


                    mean_pos_error        = position_diffs.mean()
                    max_pos_error         = position_diffs.max()
                    mean_relative_pos_error    = relative_position_diffs.mean()
                    max_relative_pos_error     = relative_position_diffs.max()

                    mean_color_error      = color_errs.mean()
                    mean_relative_color_error  = relative_color_errs.mean()

                    quality_simulation_buffer.append({
                        "mean_pos_error":      mean_pos_error,
                        "max_pos_error":       max_pos_error,
                        "mean_color_error":    mean_color_error,
                        "mean_relative_pos_error(%)":  mean_relative_pos_error,
                        "max_relative_pos_error(%)":   max_relative_pos_error,
                        "mean_relative_color_error(%)":mean_relative_color_error,
                    })

                    if quality_simulation_index is not None:

                        apply_combo_settings(simulation_combos[quality_simulation_index])
                        # dump .ply file
                        if(len(quality_simulation_buffer) == quality_simulation_buffer.maxlen):
                            try:
                                filename = f"experiments/{encoding_mode.name}_exp_{quality_simulation_index}.ply"
                                write_pointcloud_ply(
                                    filename,
                                    decoded_positions, 
                                    decoded_colors     
                                )
                                print(f"Saved PLY for exp {quality_simulation_index} → {filename}")
                            except Exception as e:
                                print(f"Error saving PLY for exp {quality_simulation_index}: {e}")
                        # write one row and advance
                        quality_simulation_index = write_quality_simulation_csv(
                            quality_simulation_buffer,
                            simulation_combos,
                            quality_simulation_index,
                            encoding_mode
                        )
                    
                    else:
                        print("QUALITY-SIM COMPLETE ✅")



                #---- CV DRAW ON SCREEN----
                now = time.perf_counter()
                if (frame_count >= 30):
                    fps = frame_count // (now - prev_time)
                    prev_time = now
                    frame_count = 0

                # draw FPS
                cv2.putText(display, f"FPS: {fps}", (cmr_clr_width -200,cmr_clr_height -30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                if recording_manager.is_active:
                    cv2.putText(
                        display,
                        "Recording...",
                        (cmr_clr_width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # initial text coordinates
                x, y0, dy = 10, 20, 20

                cv2.putText(
                    display,
                    f"Mode: {encoding_mode.name}",
                    (x, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                if encoding_mode == EncodingMode.FULL:
                    # FULL: show only the full-encode settings
                    settings_full = [
                        f"PosQuant bits: {draco_full_encoding.position_quantization_bits} / 20",
                        f"ColorQuant bits: {draco_full_encoding.color_quantization_bits} / 16",
                        f"Encoding Speed setting: {draco_full_encoding.speed_encode} / 10",
                        f"Decoding Speed setting: {draco_full_encoding.speed_decode} / 10",
                    ]

                    for i, txt in enumerate(settings_full):
                        cv2.putText(
                            display,
                            txt,
                            (x, y0 + (i+1) * dy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                    budget_txt = (
                        f"Subsample ratio: {effective_ratio*100:.1f}% @ {target_frame_rate:.1f} fps"
                        f" / {bandwidth_mbps:.1f} MB/s"
                    )
                    cv2.putText(
                        display,
                        budget_txt,
                        (x, y0 + (len(settings_full) + 2) * dy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                elif encoding_mode == EncodingMode.IMPORTANCE:
                    # IMPORTANCE: show both in-ROI and out-ROI settings
                    group1 = [
                        f"PosQuant bits (in-ROI):  {draco_roi_encoding.position_quantization_bits} / 20",
                        f"ColorQuant bits (in-ROI):{draco_roi_encoding.color_quantization_bits} / 16",
                        f"Encoding Speed setting (in-ROI):  {draco_roi_encoding.speed_encode} / 10",
                        f"Decoding Speed setting (in-ROI):  {draco_roi_encoding.speed_decode} / 10",
                    ]
                    for i, txt in enumerate(group1):
                        cv2.putText(
                            display,
                            txt,
                            (x, y0 + (i+1) * dy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                    group2 = [
                        f"PosQuant bits (out-ROI):  {draco_outside_roi_encoding.position_quantization_bits} / 20",
                        f"ColorQuant bits (out-ROI):{draco_outside_roi_encoding.color_quantization_bits} / 16",
                        f"Encoding Speed setting (out-ROI):  {draco_outside_roi_encoding.speed_encode} / 10",
                        f"Decoding Speed setting (out-ROI):  {draco_outside_roi_encoding.speed_decode} / 10",
                    ]
                    start2_y = y0 + (len(group1) + 2) * dy
                    for j, txt in enumerate(group2):
                        cv2.putText(
                            display,
                            txt,
                            (x, start2_y + j * dy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                    budget_txt = (
                        f"Subsample ratio: {effective_ratio*100:.1f}% @ {target_frame_rate:.1f} fps"
                        f" / {bandwidth_mbps:.1f} MB/s"
                    )
                    cv2.putText(
                        display,
                        budget_txt,
                        (x, start2_y + len(group2) * dy + dy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                if roi is not None:
                    cv2.rectangle(display, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 2)
            

                if visualization_mode is VizualizationMode.DEPTH:
                    depth_8u = cv2.convertScaleAbs(depth_img, alpha=255.0 / depth_img.max())
                    depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                    display = depth_colormap

                if DEBUG:
                    # Show the cursor location in the preview for quick spatial feedback.
                    cursor_text = f"Cursor point : ({cursor_position[0]}, {cursor_position[1]})"
                    cursor_anchor = (10, display.shape[0] - 10)
                    cv2.putText(
                        display,
                        cursor_text,
                        cursor_anchor,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(win_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(1) # this takes 20 ms

                if key in (ord('q'), 27):  # q or Esc
                    break
                ## Adjust settings
                elif key == ord('f'):
                    if encoding_mode == EncodingMode.NONE:
                        encoding_mode = EncodingMode.FULL
                    elif encoding_mode == EncodingMode.IMPORTANCE:
                        encoding_mode = EncodingMode.NONE
                    else:
                        encoding_mode = EncodingMode.IMPORTANCE
                elif key == ord('d'):
                    #Toggle mode (Visualization: Depth vs Color)
                    visualization_mode = (
                        visualization_mode.DEPTH
                        if visualization_mode is visualization_mode.COLOR
                        else visualization_mode.COLOR
                    )
                # HIGH IMPORTANCE (FULL, ROI) POSITION QUANTIZATION BITS
                elif key == ord('='): 
                    draco_full_encoding.position_quantization_bits = min(draco_full_encoding.position_quantization_bits+1, 20)
                elif key == ord('-'):
                    draco_full_encoding.position_quantization_bits = max(draco_full_encoding.position_quantization_bits-1, 1)
                # LOW IMPORTANCE (OUT-ROI) POSITION QUANTIZATION BITS
                elif key == ord('+'):
                    draco_outside_roi_encoding.position_quantization_bits = min(draco_outside_roi_encoding.position_quantization_bits+1, 20)
                elif key == ord('_'):
                    draco_outside_roi_encoding.position_quantization_bits = max(draco_outside_roi_encoding.position_quantization_bits-1, 1)
                # HIGH IMPORTANCE (FULL, ROI)  COLOR QUANTIZATION BITS
                elif key == ord(']'):
                    draco_full_encoding.color_quantization_bits = min(draco_full_encoding.color_quantization_bits+1, 16)
                elif key == ord('['):
                    draco_full_encoding.color_quantization_bits = max(draco_full_encoding.color_quantization_bits-1, 1)
                # LOW IMPORTANCE (OUT-ROI)  COLOR QUANTIZATION BITS
                elif key == ord('}'):
                    draco_outside_roi_encoding.color_quantization_bits = min(draco_outside_roi_encoding.color_quantization_bits+1, 16)
                elif key == ord('{'):
                    draco_outside_roi_encoding.color_quantization_bits = max(draco_outside_roi_encoding.color_quantization_bits-1, 1)
                 # HIGH IMPORTANCE (FULL, ROI) ENCODING SPEED
                elif key == ord('.'):
                    draco_full_encoding.speed_encode = min(draco_full_encoding.speed_encode+1, 10)
                elif key == ord(','):
                    draco_full_encoding.speed_encode = max(draco_full_encoding.speed_encode-1, 0)
                 # LOW IMPORTANCE (OUT-ROI)  ENCODING SPEED
                elif key == ord('>'):
                    draco_outside_roi_encoding.speed_encode = min(draco_outside_roi_encoding.speed_encode+1, 10)
                elif key == ord('<'):
                    draco_outside_roi_encoding.speed_encode = max(draco_outside_roi_encoding.speed_encode-1, 0)
                 # HIGH IMPORTANCE (FULL, ROI) DECODING SPEED
                elif key == ord('\''):
                    draco_full_encoding.speed_decode = min(draco_full_encoding.speed_decode+1, 10)
                elif key == ord(';'):
                    draco_full_encoding.speed_decode = max(draco_full_encoding.speed_decode-1, 0)
                 # LOW IMPORTANCE (OUT-ROI) DECODING  SPEED
                elif key == ord('"'):
                    draco_outside_roi_encoding.speed_decode = min(draco_outside_roi_encoding.speed_decode+1, 10)
                elif key == ord(':'):
                    draco_outside_roi_encoding.speed_decode = max(draco_outside_roi_encoding.speed_decode-1, 0)
                elif key == ord(' '):
                    # save full point cloud PLY
                    points.export_to_ply("snapshot.ply", color_frame)
                    print("Saved full point cloud to snapshot.ply")
                elif key == ord('r'):
                    target_dir = recording_manager.start_recording()
                    print(f"Recording {recording_manager.frame_target} frames into {target_dir.name}")
                elif key == ord('s'):
                    cv2.imwrite(f'./{frame_id}_out.png', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                elif key == ord('c'): # Logging button
                    if simulation:
                        # kick off the full simulation sweep
                        simulation_combos = generate_combinations(encoding_mode)
                        if simulation_combos:
                            performance_simulation_index    = 0
                            performance_simulation_buffer.clear()
                            apply_combo_settings(simulation_combos[0])
                            print(f"PERF-SIM: starting {len(simulation_combos)} combos of {encoding_mode.name}")
                        else:
                            print(f"PERF-SIM: no combos for {encoding_mode.name}, falling back to normal CSV logging")
                            write_stats_csv(
                                frame_stats_buffer,
                                encoding_mode,
                                # color & depth resolutions
                                (cmr_clr_width, cmr_clr_height),
                                (cmr_depth_width, cmr_depth_height),
                                # FULL mode params
                                draco_full_encoding.speed_encode,
                                draco_full_encoding.speed_decode,
                                draco_full_encoding.position_quantization_bits,
                                draco_full_encoding.color_quantization_bits,
                                None,
                                # IMPORTANCE mode “in” params
                                draco_roi_encoding.speed_encode,
                                draco_roi_encoding.speed_decode,
                                draco_roi_encoding.position_quantization_bits,
                                draco_roi_encoding.color_quantization_bits,
                                # IMPORTANCE mode “out” params
                                draco_outside_roi_encoding.speed_encode,
                                draco_outside_roi_encoding.speed_decode,
                                draco_outside_roi_encoding.position_quantization_bits,
                                draco_outside_roi_encoding.color_quantization_bits
                            )
                    else:
                        write_stats_csv(
                            frame_stats_buffer,
                            encoding_mode,
                            # resolutions
                            (cmr_clr_width, cmr_clr_height),
                            (cmr_depth_width, cmr_depth_height),
                            # FULL mode params
                            draco_full_encoding.speed_encode,
                            draco_full_encoding.speed_decode,
                            draco_full_encoding.position_quantization_bits,
                            draco_full_encoding.color_quantization_bits,
                            None,
                            # IMPORTANCE mode “in” params
                            draco_roi_encoding.speed_encode,
                            draco_roi_encoding.speed_decode,
                            draco_roi_encoding.position_quantization_bits,
                            draco_roi_encoding.color_quantization_bits,
                            # IMPORTANCE mode “out” params
                            draco_outside_roi_encoding.speed_encode,
                            draco_outside_roi_encoding.speed_decode,
                            draco_outside_roi_encoding.position_quantization_bits,
                            draco_outside_roi_encoding.color_quantization_bits
                        )
                elif key == ord('v'):
                    if (simulation):
                    # kick off quality simulation
                        simulation_combos = generate_combinations(encoding_mode)
                        if simulation_combos:
                            quality_simulation_index  = 0
                            quality_simulation_buffer.clear()
                            apply_combo_settings(simulation_combos[0])
                            print(f"QUAL-SIM: starting {len(simulation_combos)} combos of {encoding_mode.name}")
                        else:
                            print(f"QUAL-SIM: no combos for {encoding_mode.name}, skipping.")
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera process stopped")

def thread_worker_sam2(
        path_to_yaml, path_to_chkp, device, image_size,
        shared_frame_name,
        shared_cluster_name,
        shared_roi_name,
        frame_shape,
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event,
        ready_roi_event: mp.Event,
        predictor_ready: mp.Event) :

    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    with torch.autocast(device_type=device.__str__(), dtype=torch.bfloat16):
        predictor = sam2_camera.build_sam2_camera_predictor(
            config_file=Path(path_to_yaml).name, 
            config_path=str(Path(".", "configs", "sam2.1")),
            ckpt_path=path_to_chkp, 
            device=device,
            image_size=image_size)
        
        sf = shared_memory.SharedMemory(name=shared_frame_name)
        sc = shared_memory.SharedMemory(name=shared_cluster_name)
        sroi = shared_memory.SharedMemory(name=shared_roi_name)

        shared_frame = np.ndarray((frame_shape[0], frame_shape[1], 3), dtype=np.uint8, buffer=sf.buf)
        shared_binary_mask = np.ndarray((frame_shape[0], frame_shape[1], 1), dtype=np.bool, buffer=sc.buf)
        shared_roi = np.ndarray((1, 4), dtype=np.int32, buffer=sroi.buf)

        predictor_ready.clear()
        ready_cluster_event.clear()
        local_frame = np.zeros_like(shared_frame)
        roi_init = False
        shared_frame[:] = 0
        predictor_ready.set()

        while not stop_event.is_set():
            if ready_frame_event.is_set():
                local_frame[:] = shared_frame
                shared_binary_mask[:] = False

                if ready_roi_event.is_set():
                    roi_local = shared_roi.copy()
                    
                    roi_center = np.array([
                        [0.5 * (roi_local[0, 0] + roi_local[0, 2]), 0.5 * (roi_local[0, 1] + roi_local[0, 3])],
                        [roi_local[0, 0], roi_local[0, 1]],
                        [roi_local[0, 0], roi_local[0, 3]],
                        [roi_local[0, 2], roi_local[0, 1]],
                        [roi_local[0, 2], roi_local[0, 3]],
                    ], dtype=np.float32)
                    
                    predictor.load_first_frame(local_frame)
                    ann_frame_idx = 0
                    ann_obj_id = (1,)
                    labels = np.array([1, 1, 1, 1, 1], dtype=np.int32)
                    roi_init = True
                    _, _, out_mask_logits = predictor.add_new_prompt(frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=roi_center, labels=labels)
                    shared_binary_mask[:] = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.bool)
                    ready_roi_event.clear()
                else :
                    if roi_init:
                        _, out_mask_logits = predictor.track(local_frame)
                        shared_binary_mask[:] = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.bool)

                ready_frame_event.clear()
                ready_cluster_event.set()

def thread_worker_yoloe(
        model: YOLOE,
        device,
        shared_frame_name,
        shared_cluster_name,
        shared_roi_name,
        frame_shape,
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event,
        ready_roi_event: mp.Event,
        predictor_ready: mp.Event) :

    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    with torch.autocast(device_type=device.__str__(), dtype=torch.bfloat16):
        sf = shared_memory.SharedMemory(name=shared_frame_name)
        sc = shared_memory.SharedMemory(name=shared_cluster_name)
        sroi = shared_memory.SharedMemory(name=shared_roi_name)

        shared_frame = np.ndarray((frame_shape[0], frame_shape[1], 3), dtype=np.uint8, buffer=sf.buf)
        shared_binary_mask = np.ndarray((frame_shape[0], frame_shape[1], 1), dtype=np.bool, buffer=sc.buf)
        shared_roi = np.ndarray((1, 4), dtype=np.int32, buffer=sroi.buf)
        
        predictor_ready.clear()
        ready_cluster_event.clear()
        local_frame = np.zeros_like(shared_frame)
        roi_init = False
        cls_index = 0
        shared_frame[:] = 0
        result = model.predict(shared_frame, conf=0.1, verbose=False) #warmup
        binary_mask = torch.zeros(size=(384, 640), dtype=torch.bool, device=model.device)  #this wont work in the future...
        predictor_ready.set()

        while not stop_event.is_set():
            if ready_frame_event.is_set():
                local_frame[:] = shared_frame

                result = model.predict(local_frame, conf=0.1, verbose=False)[0]
                #class_names = result.names

                if ready_roi_event.is_set():
                    if result.boxes:
                        roi_local = shared_roi.copy()
                        roi_center = np.array([0.5 * (roi_local[0, 0] + roi_local[0, 2]), 0.5 * (roi_local[0, 1] + roi_local[0, 3])], dtype=np.float32)
                        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                        x_interval = (boxes[:, 0] <= roi_center[0]) & (roi_center[0] <= boxes[:, 2])
                        y_interval = (boxes[:, 1] <= roi_center[1]) & (roi_center[1] <= boxes[:, 3])
                        roi_isects = x_interval & y_interval

                        for roi_i, roi_isect in enumerate(roi_isects):
                            if roi_isect:
                                cls_index = int(result.boxes.cls.cpu().numpy()[roi_i])
                                roi_init = True

                    ready_roi_event.clear()

                if not roi_init:
                    shared_binary_mask[:] = False
                    if result.masks is not None:
                        binary_mask[:] = False
                        for mask in result.masks.data:
                            binary_mask |= mask > 0.5

                        binary_mask = binary_mask.type(torch.uint8)
                        out_mask = F.interpolate(binary_mask[None, None, :, :], size=(result.masks.orig_shape[0], result.masks.orig_shape[1]), mode='nearest')
                        shared_binary_mask[:] = out_mask.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.bool)
                else:
                    shared_binary_mask[:] = False

                    if result.masks is not None :
                        binary_mask[:] = False
                        classes = result.boxes.cls.cpu().numpy().astype(np.int32)

                        for box_i, cls_i in enumerate(classes):
                            if cls_i != cls_index:
                                continue

                            binary_mask |= result.masks.data[box_i, ::] > 0.5

                        binary_mask = binary_mask.type(torch.uint8)
                        out_mask = F.interpolate(binary_mask[None, None, :, :], size=(result.masks.orig_shape[0], result.masks.orig_shape[1]), mode='nearest')
                        shared_binary_mask[:] = out_mask.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.bool)
                
                ready_frame_event.clear()
                ready_cluster_event.set()

def launch_processes(server: broadcaster.ProducerServer, args, device : str) -> None:
    cmr_clr_width, cmr_clr_height = producer_cli.map_to_camera_res[args.realsense_clr_stream]
    cmr_depth_width, cmr_depth_height = producer_cli.map_to_camera_res[args.realsense_depth_stream]
    cmr_fps = args.realsense_target_fps

    simulation = args.simulation

    stop_event = mp.Event()
    ready_frame_event = mp.Event()
    ready_cluster_event = mp.Event()
    ready_roi_event = mp.Event()
    predictor_event = mp.Event()

    shm_cluster = shared_memory.SharedMemory(create=True, size=cmr_clr_width * cmr_clr_height * np.dtype(np.bool).itemsize)
    shm_frame = shared_memory.SharedMemory(create=True, size=cmr_clr_width * cmr_clr_height * 3 * np.dtype(np.uint8).itemsize)
    shm_roi = shared_memory.SharedMemory(create=True, size= 4 * np.dtype(np.int32).itemsize)

    shared_frame = np.ndarray((cmr_clr_height, cmr_clr_width, 3), dtype=np.uint8, buffer=shm_frame.buf)
    shared_cluster = np.ndarray((cmr_clr_height, cmr_clr_width, 1), dtype=np.bool, buffer=shm_cluster.buf)
    shared_roi = np.ndarray((4,), dtype=np.int32, buffer=shm_roi.buf)

    shared_frame[:] = 0
    shared_cluster[:] = False
    shared_roi[:] = 0

    if args.cluster_predictor == 'sam2':
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
        
        predictor_proc = mp.Process(target=thread_worker_sam2, 
            args=(path_to_yaml, path_to_chkp, device, args.sam2_image_size, shm_frame.name, shm_cluster.name, shm_roi.name, (cmr_clr_height, cmr_clr_width), stop_event, ready_frame_event, ready_cluster_event, ready_roi_event, predictor_event))
    elif args.cluster_predictor == 'yolo':
        predictor = None
        if args.yolo_size == 'large':
            predictor = YOLOE("yoloe-11l-seg.pt", verbose=False)
        elif args.yolo_size == 'medium':
            predictor = YOLOE("yoloe-11m-seg.pt", verbose=False)
        else:
            predictor = YOLOE("yoloe-11s-seg.pt", verbose=False)

        names = ["glasses", "shirt", "hat", "shorts"]
        predictor.set_classes(names, predictor.get_text_pe(names))

        predictor_proc = mp.Process(target=thread_worker_yoloe,
            args=(predictor, device, shm_frame.name, shm_cluster.name, shm_roi.name, (cmr_clr_height, cmr_clr_width), stop_event, ready_frame_event, ready_cluster_event, ready_roi_event, predictor_event))
    else:
        print('Failed to parse predictor')
        return

    try:
        predictor_proc.start()
        #main_thread_worker_yolo(predictor, frame_queue, result_queue, stop_event)

        while not predictor_event.is_set():
            continue

        camera_process(server, shm_frame.name, shm_cluster.name, shm_roi.name,
            cmr_clr_width, cmr_clr_height,
            cmr_depth_width, cmr_depth_height, cmr_fps,
            args.max_bandwidth_mbps, args.target_frame_rate, args.subsample_frames,
            stop_event, ready_frame_event, ready_cluster_event, ready_roi_event,
            simulation, args.record_frames)

        predictor_proc.terminate()
        predictor_proc.join()
    except KeyboardInterrupt:
        print("Stopping processes...")
        stop_event.set()

        if predictor_proc.is_alive():
            predictor_proc.terminate()

        predictor_proc.join()

    shm_cluster.close()
    shm_cluster.unlink()
    shm_frame.close()
    shm_frame.unlink()
    shm_roi.close()
    shm_roi.unlink()