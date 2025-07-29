from enum import Enum, auto
from pyexpat import model
import time
import os
import encoder

import numpy as np
import encoding as enc
import producer_cli as producer_cli

from pathlib import Path
from typing import Tuple
from broadcasting import *

import pyrealsense2 as rs
import cv2
import sam2_camera_predictor as sam2_camera
from ultralytics import YOLOE

from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.live import Live
from rich.console import Group

import statslogger as log
import multiprocessing as mp
from multiprocessing import shared_memory

import queue

from dataclasses import dataclass

from mediapipe.tasks.python.vision import (RunningMode,)

from gesture_recognition import (
    PointingGestureRecognizer,
    NormalizedBoundingBox,
    PixelBoundingBox
)

import torch
import torch.nn.functional as F

DEBUG = True

class DracoEncoder:
    def __init__(self):
        self.posQuant = 10
        self.colorQuant = 8
        self.speedEncode = 10
        self.speedDecode = 10
        self.roiWidth = 240
        self.roiHeight = 240
    def to_string(self):
        return f"Qpos:{self.posQuant:2d} Qcol:{self.colorQuant:2d} Spd:{self.speedEncode}/10 ROI:{self.roiWidth}x{self.roiHeight}"

    def encode(self, points, colors, stats: log.EncodingStats):

        stats.pts = points.shape[0]
        stats.raw_bytes = stats.pts * (3 * 4 + 3 * 1)

        buffer = encoder.encode_pointcloud(
            points,
            colors,
            self.posQuant,
            self.colorQuant,
            self.speedEncode,
            self.speedDecode
        )

        return buffer

def camera_process(
        server: bc.ProducerServer,
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
        ready_roi_event: mp.Event) :
    
    global DEBUG

    draco_executor = ProcessPoolExecutor(max_workers=2)
    
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
    pc_block = rs.pointcloud()
    frame_id = 0

    dracoAll = DracoEncoder()
    dracoROI = DracoEncoder()
    dracoOut = DracoEncoder()
    statsROI = log.EncodingStats()
    statsOut = log.EncodingStats()
    statsAll = log.EncodingStats()
    statsGeneral = log.GeneralStats()

    sf = shared_memory.SharedMemory(name=shared_frame_name)
    sc = shared_memory.SharedMemory(name=shared_cluster_name)
    sroi = shared_memory.SharedMemory(name=shared_roi_name)

    shared_frame = np.ndarray((cmr_clr_height, cmr_clr_width, 3), dtype=np.uint8, buffer=sf.buf)
    shared_cluster = np.ndarray((cmr_clr_height, cmr_clr_width, 1), dtype=np.bool, buffer=sc.buf)
    shared_roi = np.ndarray((4,), dtype=np.int32, buffer=sroi.buf)

    ready_frame_event.clear()

    try:
        while not stop_event.is_set():
            frame_id += 1
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()  
            color_frame = frames.get_color_frame()    
                        
            h, w = depth_frame.get_height(), depth_frame.get_width()

            depth_frame = depth_thresh.process(depth_frame)

            color_img = np.asanyarray(color_frame.get_data())        
            depth_img = np.asanyarray(depth_frame.get_data())

            display = color_img

            points = pc_block.calculate(depth_frame)
            pc_block.map_to(color_frame)
            
            count = points.size()
            vert_buf = points.get_vertices()
            tex_buf = points.get_texture_coordinates()
            
            verts = np.frombuffer(vert_buf, dtype=np.float32).reshape(count, 3)
            tex = np.frombuffer(tex_buf, dtype=np.float32).reshape(count, 2)
            
            # Flatten and mask arrays
            
            u = (tex[:,0] * (w-1)).astype(np.int32)
            v = (tex[:,1] * (h-1)).astype(np.int32) # 3 ms

            # Clip to valid range
            u_idx = np.clip(u, 0, w-1)
            v_idx = np.clip(v, 0, h-1) # 1ms

            pix_idx = v_idx * w + u_idx
            bgr_flat = color_img.reshape(-1,3) 
            colors = bgr_flat[pix_idx][:, ::-1]# 4ms

            # find valid points
            depth_flat = depth_img.ravel()
            valid = (depth_flat>0) & np.isfinite(verts[:,2]) & (verts[:,2]>0)

            gesture_recognizer.recognize(display, frame_id)
            roi = None

            for bounding_box_normalized in gesture_recognizer.latest_bounding_boxes:
                if bounding_box_normalized:
                    roi: np.array = bounding_box_normalized.to_pixel(display.shape[1], display.shape[0], True)
                    shared_roi[:] = roi
                    ready_roi_event.set()

            shared_frame[:] = display
            ready_frame_event.set()

            #TODO: we need to find some work todo here in order to hide latency between processes

            if ready_cluster_event.is_set():
                ready_cluster_event.clear()

            if DEBUG:
                display[shared_cluster[:, :, 0], 2] = 255

            flat_cluster = shared_cluster.flatten()
            in_roi = valid & flat_cluster
            out_roi = valid & ~flat_cluster
            pts_roi = verts[in_roi]; cols_roi = colors[in_roi]
            pts_out = verts[out_roi]; cols_out = colors[out_roi]

            futures = []

            dracoROI.posQuant = dracoAll.posQuant
            dracoROI.colorQuant = dracoAll.colorQuant
            dracoROI.speedEncode = dracoAll.speedEncode
            dracoROI.speedDecode = dracoAll.speedDecode 

            dracoOut.posQuant = dracoROI.posQuant // 2
            dracoOut.colorQuant = dracoROI.colorQuant
            dracoOut.speedEncode = dracoROI.speedEncode
            dracoOut.speedDecode = dracoROI.speedDecode
        
            if pts_roi is not None and pts_roi.shape[0] > 0: # check if non empty
                futures += [draco_executor.submit(enc._encode_chunk,
                                    pts_roi, cols_roi,
                                    dracoROI, statsROI),]
            else:
                buf_roi = b""

            if pts_out is not None:
                futures += [draco_executor.submit(enc._encode_chunk,
                                    pts_out, cols_out,
                                    dracoOut, statsOut),]
            else:
                buf_out = b""

            for future in as_completed(futures):                        
                index = futures.index(future)
                buf, stats = future.result()
                if len(futures) == 1:
                    buf_out, statsOut = buf, stats
                    statsOut.encoded_bytes = len(buf_out)
                else:
                    if index == 0:
                        buf_roi , statsROI = buf, stats
                        statsROI.encoded_bytes = len(buf_roi)
                    elif index == 1:
                        buf_out, statsOut = buf, stats
                        statsOut.encoded_bytes = len(buf_out)

            bufs = []
            if buf_roi:
                bufs.append(buf_roi)

            if buf_out:
                bufs.append(buf_out)

            count = len(bufs)
            for buf in bufs:
                server.broadcast(bytes([count]) + buf) # Prefix with byte that tells us the length

            if DEBUG:
                if roi is not None:
                    cv2.rectangle(display, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 2)
            
                cv2.imshow(win_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
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

                if ready_roi_event.is_set():
                    roi_local = shared_roi.copy()
                    roi_center = np.array([[0.5 * (roi_local[0, 0] + roi_local[0, 2]), 0.5 * (roi_local[0, 1] + roi_local[0, 3])]], dtype=np.float32)
                    predictor.load_first_frame(local_frame)
                    ann_frame_idx = 0
                    ann_obj_id = (1,)
                    labels = np.array([1], dtype=np.int32)
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

def launch_processes(server: bc.ProducerServer, args, device : str) -> None:
    cmr_clr_width = args.realsense_clr_capture_width
    cmr_clr_height = args.realsense_clr_capture_height
    cmr_depth_width = args.realsense_depth_capture_width
    cmr_depth_height = args.realsense_depth_capture_height
    cmr_fps = args.realsense_target_fps

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
            stop_event, ready_frame_event, ready_cluster_event, ready_roi_event)
        
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