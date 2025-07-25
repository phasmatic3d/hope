from enum import Enum, auto
from pyexpat import model
import time
import numpy as np

import encoder
import encoding as enc

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
from mediapipe.tasks.python.vision import (
    RunningMode,
)

from gesture_recognition import (
    PointingGestureRecognizer,
    NormalizedBoundingBox,
    PixelBoundingBox
)

import torch
import torch.nn.functional as F

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
        cmr_clr_width: int,
        cmr_clr_height: int,
        cmr_depth_width: int,
        cmr_depth_height: int,
        cmr_fps: int,
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event) :
    
    draco_executor = ProcessPoolExecutor(max_workers=2)
    DEBUG = True

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

    shared_frame = np.ndarray((cmr_clr_height, cmr_clr_width, 3), dtype=np.uint8, buffer=sf.buf)
    shared_cluster = np.ndarray((cmr_clr_height, cmr_clr_width, 1), dtype=np.bool, buffer=sc.buf)

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
            pixel_space_bounding_box = None
            roi = None

            for bounding_box_normalized in gesture_recognizer.latest_bounding_boxes:
                if bounding_box_normalized:
                    pixel_space_bounding_box: PixelBoundingBox = bounding_box_normalized.to_pixel(display.shape[1], display.shape[0])
                    roi = np.array([
                        pixel_space_bounding_box.x1,
                        pixel_space_bounding_box.y1,
                        pixel_space_bounding_box.x2,
                        pixel_space_bounding_box.y2,
                    ], dtype=np.int32)

            if roi is not None:
                cv2.rectangle(display, (roi[0], roi[1]), (roi[2], roi[3]), (0,255,0), 2)

            shared_frame[:] = display
            ready_frame_event.set()

            if ready_cluster_event.is_set():
                if DEBUG:
                    display[shared_cluster[:, :, 0], 2] = 255
                ready_cluster_event.clear()

            flat_cluster = shared_cluster.flatten()
            in_roi = valid & flat_cluster
            out_roi = valid & ~flat_cluster
            pts_roi = verts[in_roi]; cols_roi = colors[in_roi]
            pts_out = verts[out_roi]; cols_out = colors[out_roi]

            if True:
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
                cv2.imshow(win_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera process stopped")

def main_thread_worker_sam(
        predictor: sam2_camera.SAM2CameraPredictor,
        frame_queue: mp.Queue,
        result_queue: mp.Queue,
        stop_event: mp.Queue) :

    win_name = "RealSense Color"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    if_sam_init = False
    out_mask_logits = None
    frame_data = None

    while not stop_event.is_set():
        try :
            cur_frame_data = frame_queue.get(block=True)
            frame_data = cur_frame_data
        except queue.Empty:
            pass

        if frame_data is None:
            continue

        frame = frame_data.frame
   
        if frame_data.roi is not None:
            cv2.rectangle(frame, (frame_data.roi[0], frame_data.roi[1]), (frame_data.roi[2], frame_data.roi[3]), (0,255,0), 2)

        if not if_sam_init and frame_data.roi is not None:
            if_sam_init = True
            predictor.load_first_frame(frame)

            ann_frame_idx = 0
            ann_obj_id = (1,)
            labels = np.array([1], dtype=np.int32)
            points = np.array([[0.5 * (frame_data.roi[0] + frame_data.roi[2]), 0.5 * (frame_data.roi[1] + frame_data.roi[3])]], dtype=np.float32)

            _, _, out_mask_logits = predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
        elif if_sam_init :
            _, out_mask_logits = predictor.track(frame)

        if out_mask_logits is not None:
            out_mask_logits = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy()
            out_mask = out_mask_logits.astype(np.uint8)
            segmentaion_mask = np.zeros_like(frame, dtype=np.uint8)
            segmentaion_mask[:, :, 0] = out_mask[:, :, 0] * 255
            frame = cv2.addWeighted(frame, 1, segmentaion_mask, 1, 0)
        
            #try:
            #    resultData = ProcessedData(out_mask_logits, time.time())
            #    result_queue.put(resultData)
            #except queue.Full:
            #    print("Frame queue is full, skipping frame")

            out_mask_logits = None

        cv2.imshow(win_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def main_thread_worker_yolo(
        model: YOLOE,
        shared_frame_name,
        shared_cluster_name,
        frame_shape,
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event) :

    sf = shared_memory.SharedMemory(name=shared_frame_name)
    sc = shared_memory.SharedMemory(name=shared_cluster_name)

    shared_frame = np.ndarray((frame_shape[0], frame_shape[1], 3), dtype=np.uint8, buffer=sf.buf)
    shared_cluster = np.ndarray((frame_shape[0], frame_shape[1], 1), dtype=np.bool, buffer=sc.buf)
    ready_cluster_event.clear()
    local_frame = np.zeros_like(shared_frame)

    while not stop_event.is_set():
        if ready_frame_event.is_set():
            local_frame[:] = shared_frame

            #perf = time.perf_counter()
            result = model.predict(local_frame, conf=0.1, verbose=False)[0]

            #boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
            #cls = result.boxes.cls.cpu().numpy()
            #binary_mask = np.zeros_like(frame, dtype=np.bool, buffer=shm.buf)
            shared_cluster[:] = False

            if result.masks is not None:
                binary_mask = torch.zeros(size=(result.masks.data.shape[1], result.masks.data.shape[2]), dtype=torch.bool, device=result.masks.data.device)
                for mask in result.masks.data:
                    binary_mask |= mask > 0.5

                binary_mask = binary_mask.type(torch.uint8)
                binary_mask = F.interpolate(binary_mask[None, None, :, :], size=(result.masks.orig_shape[0], result.masks.orig_shape[1]), mode='nearest')
                shared_cluster[:] = binary_mask.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.bool)

                #for (box, c) in zip(boxes, cls):
                #    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                #    cv2.putText(frame, model.names[c], (box[0] - 1, box[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            #perf = (time.perf_counter() - perf) * 100
            #print(f'time to finish {perf}')
            
            ready_frame_event.clear()
            ready_cluster_event.set()

def launch_processes(server: bc.ProducerServer,
        cmr_clr_width: int,
        cmr_clr_height: int,
        cmr_depth_width: int,
        cmr_depth_height: int,
        cmr_fps: int,
        path_to_yaml: str,
        path_to_chkp: str,
        device: str,
        image_size: int) -> None:
    
    stop_event = mp.Event()
    ready_frame_event = mp.Event()
    ready_cluster_event = mp.Event()

    shm_cluster = shared_memory.SharedMemory(create=True, size=cmr_clr_width * cmr_clr_height * np.dtype(np.bool).itemsize)
    shm_frame = shared_memory.SharedMemory(create=True, size=cmr_clr_width * cmr_clr_height * 3 * np.dtype(np.uint8).itemsize)
    
    shared_frame = np.ndarray((cmr_clr_height, cmr_clr_width, 3), dtype=np.uint8, buffer=shm_frame.buf)
    shared_cluster = np.ndarray((cmr_clr_height, cmr_clr_width, 1), dtype=np.bool, buffer=shm_cluster.buf)

    shared_frame[:] = 0
    shared_cluster[:] = False

    if False:
        predictor = sam2_camera.build_sam2_camera_predictor(
            config_file=Path(path_to_yaml).name, 
            config_path=str(Path(".", "configs", "sam2.1")),
            ckpt_path=path_to_chkp, 
            device=device,
            image_size=image_size)
    else:
        predictor = YOLOE("yoloe-11l-seg.pt", verbose=False)
        names = ["glasses", "shirt", "hat", "shorts"]
        predictor.set_classes(names, predictor.get_text_pe(names))

    predictor_proc = mp.Process(target=main_thread_worker_yolo, 
        args=(predictor, shm_frame.name, shm_cluster.name, (cmr_clr_height, cmr_clr_width), stop_event, ready_frame_event, ready_cluster_event))

    try:
        predictor_proc.start()
        #main_thread_worker_yolo(predictor, frame_queue, result_queue, stop_event)

        camera_process(server, shm_frame.name, shm_cluster.name, cmr_clr_width, cmr_clr_height,
            cmr_depth_width, cmr_depth_height, cmr_fps,
            stop_event, ready_frame_event, ready_cluster_event)
        
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