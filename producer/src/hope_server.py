from enum import Enum, auto
from pyexpat import model
import time
import os

import pandas as pd

import numpy as np
import producer_cli as producer_cli

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
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event,
        ready_roi_event: mp.Event,
        simulation: bool) :
    
    def apply_combo_settings(combo): # Helper function for setting a combination for simulations
        if encoding_mode == EncodingMode.FULL:
            draco_full_encoding.position_quantization_bits = combo["pos_bits"]
            draco_full_encoding.color_quantization_bits    = combo["pos_bits"]
            draco_full_encoding.speed_encode               = combo["speed"]
            draco_full_encoding.speed_decode               = combo["speed"]
        elif encoding_mode == EncodingMode.IMPORTANCE:
            draco_roi_encoding.position_quantization_bits        = combo["pos_bits_in"]
            draco_roi_encoding.color_quantization_bits           = combo["pos_bits_in"]
            draco_roi_encoding.speed_encode                      = combo["speed_in"]
            draco_roi_encoding.speed_decode                      = combo["speed_in"]
            draco_outside_roi_encoding.position_quantization_bits = combo["pos_bits_out"]
            draco_outside_roi_encoding.color_quantization_bits    = combo["pos_bits_out"]
            draco_outside_roi_encoding.speed_encode               = combo["speed_out"]
            draco_outside_roi_encoding.speed_decode               = combo["speed_out"]
            active_layers[:] = combo["layers"]
    
    global DEBUG

    # Hyperparameters to move to argparse
    visualization_mode = VizualizationMode.COLOR
    encoding_mode      = EncodingMode.FULL

    frame_stats_buffer = deque(maxlen=30) # buffer for dynamic stats logging (CSV)

    # simulation settings for simulation csv logging
    simulation_combos   = []
    simulation_index    = None
    simulation_buffer   = deque(maxlen=30)

    thread_executor = ThreadPoolExecutor(max_workers=2)
    
    if DEBUG:
        win_name = "RealSense vis"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

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
    
    min_dist = 0.1
    max_dist = 1.3
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


    # --- SUBSAMPLING LAYERS SETUP ---
    # layer 0 = 60%, layer 1 = 15%, layer 2 = 25%
    sampling_layers = [0.60, 0.15, 0.25]
    active_layers   = [True,  True,  True]

    broadcast_round = 0 # Keep track of broadcasting round to query cpp csv for logging
    batch = 0
    
    try:
        while not stop_event.is_set():
            # Sync hyperparameter changes
            draco_roi_encoding.position_quantization_bits = draco_full_encoding.position_quantization_bits
            draco_roi_encoding.color_quantization_bits    = draco_full_encoding.color_quantization_bits
            draco_roi_encoding.speed_encode               = draco_full_encoding.speed_encode
            draco_roi_encoding.speed_decode               = draco_full_encoding.speed_decode 


            frame_id += 1
            frame_count += 1
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

            if (encoding_mode == EncodingMode.NONE):
                pipeline_stats.data_preparation_ms = (time.perf_counter() - pipeline_stats.data_preparation_ms) * 1000 #prep end

       
                raw_points = vertices    # shape: (N,3) float32
                raw_cols = colors     # shape: (N,3) uint8



                if(raw_points.size > 0):
                    payload = bytes([0]) + raw_points.tobytes() + raw_cols.tobytes()
                    server.broadcast(payload)

                entry = server.wait_for_entry(broadcast_round)
                if entry:
                    broadcast_round += 1
                    pipeline_stats.approximate_rtt_ms = entry.approximate_rtt_ms

            else:

                build_valid_points_start_time = time.perf_counter()
                # find valid points
                depth_flat = depth_img.ravel()
                # Build a (H*W,) boolean mask where True means:
                #  1) the depth sensor saw something (depth_flat > 0),
                #  2) the reconstructed Z coordinate is a finite number,
                valid = (
                    (depth_flat > 0)                         # non-zero depth reading
                    & np.isfinite(vertices[:, 2])      # Z isn’t NaN or ±Inf
                )

                pipeline_stats.build_valid_points_ms = (time.perf_counter() - build_valid_points_start_time) * 1000
                
                # --- SUBSAMPLING ---
                subsampling_time_start = time.perf_counter()
                effective_ratio = sum(r for r, on in zip(sampling_layers, active_layers) if on)

                # roll one random array and reject ~ (1‑effective_ratio) of valid points
                rnd = np.random.rand(valid.shape[0])
                subsample_mask = rnd < effective_ratio

                # Add the subsampling mask to the valid points
                #valid &= subsample_mask # TODO: Maybe do not subsample the entire point cloud (maybe this has use in IMPORTANCE mode)
                pipeline_stats.subsampling_ms = (time.perf_counter() - subsampling_time_start ) * 1000
                
                if(encoding_mode == EncodingMode.IMPORTANCE):
                    # ---GESTURE RECOGNITION---
                    gesture_recognizer.recognize(display, frame_id)

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
                    in_roi = valid & flat_cluster
                    out_roi = valid & ~in_roi & subsample_mask
                    points_in_roi = vertices[in_roi]; colors_in_roi = colors[in_roi]
                    points_out_roi = vertices[out_roi]; colors_out_roi = colors[out_roi]
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
                    points_full_frame = vertices[valid]
                    colors_full_frame = colors[valid]
                    compression_full_stats.masking_ms = (time.perf_counter() - compression_full_stats.masking_ms ) * 1000
                

                    # Encode entire valid cloud
                    pipeline_stats.data_preparation_ms = (time.perf_counter() - pipeline_stats.data_preparation_ms) * 1000 #prep end
                    if(points_full_frame.any()):
                        buffer_full = draco_full_encoding.encode(points_full_frame, colors_full_frame)
                        server.broadcast(bytes([1]) + buffer_full) # prefix with single byte to understand that we are sending one buffer
                    
                    entry = server.wait_for_entry(broadcast_round)
                    if entry:
                        broadcast_round += 1
                       
                        
            

            # Logging and display
            if DEBUG:
                # Logging
                if(entry): # If broadcasting
                    frame_stats_buffer.append({
                        "frame_preparation_ms":         pipeline_stats.frame_preparation_ms,
                        "data_preparation_ms":         pipeline_stats.data_preparation_ms,
                        "multiprocessing_compression_ms": pipeline_stats.multiprocessing_compression_ms,
                        "one_way_ms":                       entry.one_way_ms,
                        "decode_ms":                        entry.pure_decode_ms,
                        "geometry_upload_ms":                entry.pure_geometry_upload_ms,
                        "full_encode_ms":              compression_full_stats.compression_ms,
                        "roi_encode_ms":               compression_roi_stats.compression_ms,
                        "outside_encode_ms":           compression_out_stats.compression_ms,
                        "num_points":                  int(num_points),
                        "full_points":                  int(compression_full_stats.number_of_points),
                        "in_roi_points":                int(compression_roi_stats.number_of_points),
                        "out_roi_points":               int(compression_out_stats.number_of_points)
                    })

                    if simulation_index is not None: # only if simulation is running
                        simulation_buffer.append({
                            "frame_preparation_ms":         pipeline_stats.frame_preparation_ms,
                            "data_preparation_ms":          pipeline_stats.data_preparation_ms,
                            "multiprocessing_compression_ms": pipeline_stats.multiprocessing_compression_ms,
                            "one_way_ms":                       entry.one_way_ms,
                            "decode_ms":                        entry.pure_decode_ms,
                            "geometry_upload_ms":                  entry.pure_geometry_upload_ms,
                            "full_encode_ms":               compression_full_stats.compression_ms,
                            "roi_encode_ms":                compression_roi_stats.compression_ms,
                            "outside_encode_ms":            compression_out_stats.compression_ms,
                            "num_points":                   int(num_points),
                            "full_points":                  int(compression_full_stats.number_of_points),
                            "in_roi_points":                int(compression_roi_stats.number_of_points),
                            "out_roi_points":               int(compression_out_stats.number_of_points),
                        })

                        simulation_index = write_simulation_csv(
                            simulation_buffer,
                            simulation_combos,
                            simulation_index,
                            encoding_mode,
                            (cmr_clr_width, cmr_clr_height),
                            (cmr_depth_width, cmr_depth_height),
                        )
                        if simulation_index is not None:
                            apply_combo_settings(simulation_combos[simulation_index])
                        else:
                            print("SIMULATION COMPLETE ✅")

                now = time.perf_counter()
                if (frame_count >= 30):
                    fps = frame_count // (now - prev_time)
                    prev_time = now
                    frame_count = 0




                #---- CV DRAW ON SCREEN----

                # draw FPS
                cv2.putText(display, f"FPS: {fps}", (cmr_clr_width -200,cmr_clr_height -30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

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
                        f"Speed setting: {draco_full_encoding.speed_encode} / 10",
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

                    # draw sampling layers just below those
                    layer_str = "   ".join(
                        f"L{i}{' ON' if active_layers[i] else ' OFF'}({int(sampling_layers[i]*100)}%)"
                        for i in range(len(sampling_layers))
                    )
                    layers_y = y0 + (len(settings_full) + 2) * dy
                    cv2.putText(
                        display,
                        layer_str,
                        (x, layers_y),
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
                        f"Speed setting (in-ROI):  {draco_roi_encoding.speed_encode} / 10",
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
                        f"Speed setting (out-ROI):  {draco_outside_roi_encoding.speed_encode} / 10",
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

                    layer_str = "   ".join(
                        f"L{i}{' ON' if active_layers[i] else ' OFF'}({int(sampling_layers[i]*100)}%)"
                        for i in range(len(sampling_layers))
                    )
                    layers_y = start2_y + len(group2) * dy + dy
                    cv2.putText(
                        display,
                        layer_str,
                        (x, layers_y),
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
                # HIGH IMPORTANCE (FULL, ROI)
                elif key == ord('='):
                    draco_full_encoding.position_quantization_bits = min(draco_full_encoding.position_quantization_bits+1, 20)
                elif key == ord('-'):
                    draco_full_encoding.position_quantization_bits = max(draco_full_encoding.position_quantization_bits-1, 1)
                # LOW IMPORTANCE (OUT-ROI)    
                elif key == ord('+'):
                    draco_outside_roi_encoding.position_quantization_bits = min(draco_outside_roi_encoding.position_quantization_bits+1, 20)
                elif key == ord('_'):
                    draco_outside_roi_encoding.position_quantization_bits = max(draco_outside_roi_encoding.position_quantization_bits-1, 1)
                # HIGH IMPORTANCE (FULL, ROI)
                elif key == ord(']'):
                    draco_full_encoding.color_quantization_bits = min(draco_full_encoding.color_quantization_bits+1, 16)
                elif key == ord('['):
                    draco_full_encoding.color_quantization_bits = max(draco_full_encoding.color_quantization_bits-1, 1)
                # LOW IMPORTANCE (OUT-ROI)    
                elif key == ord('}'):
                    draco_outside_roi_encoding.color_quantization_bits = min(draco_outside_roi_encoding.color_quantization_bits+1, 16)
                elif key == ord('{'):
                    draco_outside_roi_encoding.color_quantization_bits = max(draco_outside_roi_encoding.color_quantization_bits-1, 1)
                 # HIGH IMPORTANCE (FULL, ROI)
                elif key == ord('.'):
                    draco_full_encoding.speed_encode = min(draco_full_encoding.speed_encode+1, 10)
                    draco_full_encoding.speed_decode = draco_full_encoding.speed_decode
                elif key == ord(','):
                    draco_full_encoding.speed_encode = max(draco_full_encoding.speed_encode-1, 0)
                    draco_full_encoding.speed_decode = draco_full_encoding.speed_decode
                 # LOW IMPORTANCE (OUT-ROI)    
                elif key == ord('>'):
                    draco_outside_roi_encoding.speed_encode = min(draco_outside_roi_encoding.speed_encode+1, 10)
                    draco_outside_roi_encoding.speed_decode = draco_outside_roi_encoding.speed_decode
                elif key == ord('<'):
                    draco_outside_roi_encoding.speed_encode = max(draco_outside_roi_encoding.speed_encode-1, 0)
                    draco_outside_roi_encoding.speed_decode = draco_outside_roi_encoding.speed_decode
                elif key == ord(' '):
                    # save full point cloud PLY
                    points.export_to_ply("snapshot.ply", color_frame)
                    print("Saved full point cloud to snapshot.ply")
                # LAYERS TOGGLE
                elif key == ord('1'):
                    active_layers[0] = not active_layers[0]
                elif key == ord('2'):
                    active_layers[1] = not active_layers[1]
                elif key == ord('3'):
                    active_layers[2] = not active_layers[2]
                elif key == ord('c'): # Logging button
                    if simulation:
                        # kick off the full simulation sweep
                        simulation_combos = generate_combinations(encoding_mode)
                        if simulation_combos:
                            simulation_index    = 0
                            simulation_buffer.clear()
                            apply_combo_settings(simulation_combos[0])
                            print(f"SIM: starting {len(simulation_combos)} combos of {encoding_mode.name}")
                        else:
                            print(f"SIM: no combos for {encoding_mode.name}, falling back to normal CSV logging")
                            write_stats_csv(
                                frame_stats_buffer,
                                encoding_mode,
                                # color & depth resolutions
                                (cmr_clr_width, cmr_clr_height),
                                (cmr_depth_width, cmr_depth_height),
                                # FULL mode params
                                draco_full_encoding.speed_encode,
                                draco_full_encoding.position_quantization_bits,
                                active_layers,
                                # IMPORTANCE mode “in” params
                                draco_roi_encoding.speed_encode,
                                draco_roi_encoding.position_quantization_bits,
                                # IMPORTANCE mode “out” params
                                draco_outside_roi_encoding.speed_encode,
                                draco_outside_roi_encoding.position_quantization_bits
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
                            draco_full_encoding.position_quantization_bits,
                            active_layers,
                            # IMPORTANCE mode “in” params
                            draco_roi_encoding.speed_encode,
                            draco_roi_encoding.position_quantization_bits,
                            # IMPORTANCE mode “out” params
                            draco_outside_roi_encoding.speed_encode,
                            draco_outside_roi_encoding.position_quantization_bits
                        )
                
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
    cmr_clr_width = args.realsense_clr_capture_width
    cmr_clr_height = args.realsense_clr_capture_height
    cmr_depth_width = args.realsense_depth_capture_width
    cmr_depth_height = args.realsense_depth_capture_height
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
            stop_event, ready_frame_event, ready_cluster_event, ready_roi_event,
            simulation)
        
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