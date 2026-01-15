from enum import Enum, auto
import time
import os
import mediapipe as mediap
import cupy as cp
import numpy as np
import producer_cli as producer_cli

from pathlib import Path
from mediapipe.framework.formats import landmark_pb2

from cuda_quantizer import CudaQuantizer, EncodingMode

import pyrealsense2 as rs
import cv2
import sam2_camera_predictor as sam2_camera
from ultralytics import YOLOE

import concurrent.futures

from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor
)

import multiprocessing as mp

from mediapipe.tasks.python.vision import (RunningMode,)

from point_gesture_recognizer import PointingGestureRecognizer

import torch
import torch.nn.functional as F

class VisualizationMode(Enum):
    COLOR = auto()
    DEPTH = auto()

def open_ipc_array(handle, shape, dtype):
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    dev_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle, cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess)
    mem = cp.cuda.UnownedMemory(dev_ptr, size, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(shape, dtype=dtype, memptr=memptr)

def camera_process(
        server,
        d_shared_frame,
        d_shared_cluster,
        d_shared_roi,
        cmr_clr_width: int,
        cmr_clr_height: int,
        cmr_depth_width: int,
        cmr_depth_height: int,
        cmr_fps: int,
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event,
        ready_roi_event: mp.Event,
        point_cloud_budget : int,
        min_depth_meter : float,
        max_depth_meter : float,
        debug : bool) :

    visualization_mode = VisualizationMode.COLOR
    quantizer = CudaQuantizer()
    thread_executor = ThreadPoolExecutor(max_workers=3)
    
    if debug:
        win_name = "RealSense vis"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, cmr_depth_width, cmr_depth_height, rs.format.z16, cmr_fps)
    cfg.enable_stream(rs.stream.color, cmr_clr_width, cmr_clr_height, rs.format.rgb8, cmr_fps)
    pipeline.start(cfg)

    profile = pipeline.get_active_profile()
    video_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
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
        delay_frames=15,
        debug=debug)
    
    depth_thresh = rs.threshold_filter(min_depth_meter, max_depth_meter)
    align = rs.align(rs.stream.color)
    rpc = rs.pointcloud()

    frame_id = 0
    frame_count = 0
    roi = None
    d_prev_cluster = cp.zeros_like(d_shared_cluster)
    
    stream_in = cp.cuda.Stream(non_blocking=True)
    stream_med = cp.cuda.Stream(non_blocking=True)
    stream_out = cp.cuda.Stream(non_blocking=True)
    
    def gpu_subsample(d_indices, budget):
        if d_indices.size <= budget: return d_indices
        return cp.random.choice(d_indices, budget, replace=False)
    
    def fast_subsample(d_indices, budget):
        n = d_indices.size
        if n <= budget:
            return d_indices

        idx_to_keep = cp.linspace(0, n - 1, num=budget, dtype=cp.int32)
        return d_indices[idx_to_keep]

    def allocate_budgets(n_in, n_mid, n_out, total_budget):
        if n_in >= total_budget:
            return total_budget, 0, 0
        
        remaining = total_budget - n_in
        if n_mid >= remaining:
            return n_in, remaining, 0
        
        remaining -= n_mid
        return n_in, n_mid, min(n_out, remaining)
    
    pinned_mem_high = cp.cuda.alloc_pinned_memory(point_cloud_budget * 15)
    pinned_mem_med  = cp.cuda.alloc_pinned_memory(point_cloud_budget * 15)
    pinned_mem_low  = cp.cuda.alloc_pinned_memory(point_cloud_budget * 15)

    pinned_np_high = np.frombuffer(pinned_mem_high, dtype=np.uint8)
    pinned_np_med = np.frombuffer(pinned_mem_med, dtype=np.uint8)
    pinned_np_low = np.frombuffer(pinned_mem_low, dtype=np.uint8)
    
    try:
        while not stop_event.is_set():
            ms_now = time.perf_counter()
            frame_count += 1
            frame_id += 1
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()  
            color_frame = frames.get_color_frame()    
            
            if not depth_frame or not color_frame:
                continue

            depth_frame = depth_thresh.process(depth_frame)
            
            color_img = np.asanyarray(color_frame.get_data())   
            depth_img = np.asanyarray(depth_frame.get_data())
            
            color_img_display = None
            
            if debug:
                color_img_display = color_img.copy()

            gesture_recognizer.recognize(color_img, frame_id)

            for bounding_box_normalized in gesture_recognizer.latest_bounding_boxes:
                if bounding_box_normalized:
                    roi = bounding_box_normalized.to_pixel(color_img.shape[1], color_img.shape[0], True)
                    
                    d_shared_roi[:] = cp.asarray(roi).reshape(1, 4)
                    cp.cuda.Stream.null.synchronize()
                    ready_roi_event.set()

            d_shared_frame[:] = cp.asarray(color_img)
            cp.cuda.Stream.null.synchronize()
            ready_frame_event.set()

            if ready_cluster_event.is_set():
                d_prev_cluster[:] = d_shared_cluster
                cp.cuda.Stream.null.synchronize() 
                ready_cluster_event.clear()

            rpc.map_to(color_frame)
            points = rpc.calculate(depth_frame)
            num_points = points.size()

            v_buf = points.get_vertices()
            t_buf = points.get_texture_coordinates()

            d_vertices = cp.frombuffer(v_buf, dtype=np.float32).reshape(num_points, 3)
            d_texcoords = cp.frombuffer(t_buf, dtype=np.float32).reshape(num_points, 2)
            d_depth_img = cp.asarray(depth_img) 
            
            d_pixel_indices = (
                (d_texcoords[:, 1] * (cmr_clr_height - 1)).astype(cp.int32) * cmr_clr_width +
                (d_texcoords[:, 0] * (cmr_clr_width - 1)).astype(cp.int32))

            d_flat_colors = d_shared_frame.reshape(-1, 3)
            d_colors = d_flat_colors[d_pixel_indices] 

            d_depth_flat = d_depth_img.ravel()
            d_valid_idx = cp.flatnonzero(d_depth_flat > 0)
            #d_prev_cluster_flat = d_prev_cluster.ravel()

            d_cluster_values = d_prev_cluster.ravel()[d_valid_idx]
            d_in_mask = (d_cluster_values == 1)
            d_mid_mask = (d_cluster_values == 2)
            d_out_mask = (d_cluster_values == 0)

            n_in = cp.count_nonzero(d_in_mask)
            n_mid = cp.count_nonzero(d_mid_mask)
            n_out = cp.count_nonzero(d_out_mask)

            budget_in, budget_mid, budget_out = allocate_budgets(
                int(n_in), int(n_mid), int(n_out), point_cloud_budget)
            
            broadcast_buffers = {}
            num_chunks = 0
            futures_list: list[concurrent.futures.Future] = []
            byte_offset = 0
            frame_id_byte = frame_count % 255
            
            #t = time.perf_counter()

            with stream_in:
                if budget_in > 0:
                    d_in_idx = d_valid_idx[d_in_mask]
                    if d_in_idx.size > budget_in:
                        d_in_idx = fast_subsample(d_in_idx, budget_in)
                    
                    if d_in_idx.size > 0:
                        res_view = quantizer.encode(
                            stream_in,
                            EncodingMode.HIGH, 
                            d_vertices[d_in_idx], 
                            d_colors[d_in_idx],
                            pinned_np_high)
                        broadcast_buffers[EncodingMode.HIGH] = (stream_in, d_in_idx.size, res_view)
                        num_chunks += 1
                     
            with stream_med:
                if budget_mid > 0:
                    d_mid_idx = d_valid_idx[d_mid_mask]
                    if d_mid_idx.size > budget_mid:
                        d_mid_idx = fast_subsample(d_mid_idx, budget_mid)
                    
                    if d_mid_idx.size > 0:
                        res_view = quantizer.encode(
                            stream_med,
                            EncodingMode.MED, 
                            d_vertices[d_mid_idx], 
                            d_colors[d_mid_idx],
                            pinned_np_med)
                        broadcast_buffers[EncodingMode.MED] = (stream_med, d_mid_idx.size, res_view)
                        num_chunks += 1

            with stream_out:
                if budget_out > 0:
                    d_out_idx = d_valid_idx[d_out_mask]
                    if d_out_idx.size > budget_out:
                        d_out_idx = fast_subsample(d_out_idx, budget_out)
                    
                    if d_out_idx.size > 0:
                        res_view = quantizer.encode(
                            stream_out, 
                            EncodingMode.LOW, 
                            d_vertices[d_out_idx], 
                            d_colors[d_out_idx],
                            pinned_np_low)
                        broadcast_buffers[EncodingMode.LOW] = (stream_out, d_out_idx.size, res_view)
                        num_chunks += 1
        
            for mode, (stream, size, buffer_view)  in broadcast_buffers.items() :
                stream.synchronize()

                #buffer_size = quantizer.estimate_buffer_size(mode, points_i.shape[0])
                #print(f'mode: {mode} - size: {size} buffer: {len(buffer) / 1024}')
                buffer_bytes = buffer_view.tobytes()
                
                header = (
                        num_chunks.to_bytes(1, 'little') + 
                        frame_id_byte.to_bytes(1, 'little') + 
                        byte_offset.to_bytes(4, 'little'))

                futures_list += [thread_executor.submit(server.broadcast, header + buffer_bytes),]
                byte_offset += len(buffer_bytes)
            
            concurrent.futures.wait(futures_list)
            #print(f'->{(time.perf_counter() - t) * 1000}')
            
            if debug:
                if d_prev_cluster is not None:
                    cluster_cpu = cp.asnumpy(d_prev_cluster)
                    color_img_display[cluster_cpu[:, :, 0] == 1, 2] = 255
                    color_img_display[cluster_cpu[:, :, 0] == 2, 1] = 255
                    
                frame_s = (time.perf_counter() - ms_now)
                fps_text = f"{int(1 / frame_s)}fps / {frame_s * 1000:.2f}ms"
                cv2.putText(color_img_display, fps_text, (cmr_clr_width - 350, cmr_clr_height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if roi is not None:
                    cv2.rectangle(color_img_display, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                    roi = None
                    
                def ensure_landmark_list(data):
                    if isinstance(data, landmark_pb2.NormalizedLandmarkList):
                        return data
                    elif isinstance(data, list):
                        return landmark_pb2.NormalizedLandmarkList(
                            landmark=[landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in data]
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
                                image=color_img_display,
                                landmark_list=hand_landmark,
                                connections=mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2))
                    gesture_recognizer.cb_result = None

                if visualization_mode is VisualizationMode.DEPTH:
                    if depth_img.max() > 0:
                        depth_8u = cv2.convertScaleAbs(depth_img, alpha=255.0 / depth_img.max())
                        depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                        color_img_display = depth_colormap

                cv2.imshow(win_name, cv2.cvtColor(cv2.resize(color_img_display, (1280, 720)), cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1)
                
                if key in (ord('q'), 27): 
                    break
                elif key == ord('d'):
                    visualization_mode = (
                        VisualizationMode.DEPTH if visualization_mode == VisualizationMode.COLOR else VisualizationMode.COLOR
                    )
            
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera process stopped")

def thread_worker_sam2(
        path_to_yaml, path_to_chkp, device, image_size,
        ipc_queue: mp.Queue, 
        frame_shape,
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event,
        ready_roi_event: mp.Event,
        predictor_ready: mp.Event) :

    cp.cuda.Device(0).use()
    
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
        
        h_frame, h_cluster, h_roi = ipc_queue.get()
        d_frame = open_ipc_array(h_frame, (frame_shape[0], frame_shape[1], 3), cp.uint8)
        d_cluster = open_ipc_array(h_cluster, (frame_shape[0], frame_shape[1], 1), cp.uint8)
        d_roi = open_ipc_array(h_roi, (1, 4), cp.int32)

        predictor_ready.clear()
        ready_cluster_event.clear()
        roi_init = False
        d_cluster[:] = 0
        predictor_ready.set()
        
        def updateMask(arr):
            logits = arr[0]
            mask1_t = (logits > 0.0)
            pooled_windows = F.max_pool2d(mask1_t.float(), kernel_size=32, stride=1, ceil_mode=True)
            expanded_mask_t = F.interpolate(pooled_windows.unsqueeze(0), size=logits.shape[-2:], mode='nearest').squeeze(0)
            mask2_t = (expanded_mask_t > 0.5) & (~mask1_t)
            
            final_mask = torch.zeros_like(mask1_t, dtype=torch.uint8)
            final_mask[mask1_t] = 1
            final_mask[mask2_t] = 2
            
            d_cluster[:] = cp.asarray(final_mask.permute(1, 2, 0).contiguous())

        while not stop_event.is_set():
            if ready_frame_event.is_set():
                d_cluster[:] = 0

                if ready_roi_event.is_set():
                    roi_local = torch.as_tensor(d_roi, device=device)
                    roi_center = torch.tensor([
                        [0.5 * (roi_local[0, 0] + roi_local[0, 2]), 0.5 * (roi_local[0, 1] + roi_local[0, 3])],
                        [roi_local[0, 0], roi_local[0, 1]],
                        [roi_local[0, 0], roi_local[0, 3]],
                        [roi_local[0, 2], roi_local[0, 1]],
                        [roi_local[0, 2], roi_local[0, 3]],
                    ], dtype=torch.float32)
                    
                    predictor.load_first_frame(torch.as_tensor(d_frame, device=device).permute(2, 0, 1))
                    
                    ann_frame_idx = 0
                    ann_obj_id = (1,)
                    labels = np.array([1, 1, 1, 1, 1], dtype=np.int32)
                    roi_init = True
                    
                    _, _, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=roi_center,
                        labels=labels)

                    updateMask(out_mask_logits)
                    ready_roi_event.clear()
                else :
                    if roi_init:
                        _, out_mask_logits = predictor.track(torch.as_tensor(d_frame, device=device).permute(2, 0, 1))
                        updateMask(out_mask_logits)
                
                cp.cuda.Stream.null.synchronize()
                ready_frame_event.clear()
                ready_cluster_event.set()

def thread_worker_yoloe(
        model: YOLOE,
        device,
        ipc_queue: mp.Queue,
        frame_shape,
        stop_event: mp.Event,
        ready_frame_event: mp.Event,
        ready_cluster_event: mp.Event,
        ready_roi_event: mp.Event,
        predictor_ready: mp.Event) :

    cp.cuda.Device(0).use()

    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def update_clusters(mask_input, target_shape):
        if mask_input.dim() == 2:
            mask_in = mask_input[None, None, :, :].float()
        else:
            mask_in = mask_input.float()

        mask1_t = F.interpolate(mask_in, size=target_shape, mode='nearest')
        pooled = F.max_pool2d(mask1_t, kernel_size=32, stride=1, ceil_mode=True)
        expanded = F.interpolate(pooled, size=target_shape, mode='nearest')
        m1 = (mask1_t > 0.5)
        m2 = (expanded > 0.5) & (~m1)
        
        final_map = torch.zeros_like(m1, dtype=torch.uint8)
        final_map[m1] = 1
        final_map[m2] = 2
        
        d_cluster[:] = cp.asarray(final_map.permute(0, 2, 3, 1).squeeze(0).contiguous())
            
    with torch.autocast(device_type=device.__str__(), dtype=torch.bfloat16):
        h_frame, h_cluster, h_roi = ipc_queue.get()
        d_frame = open_ipc_array(h_frame, (frame_shape[0], frame_shape[1], 3), cp.uint8)
        d_cluster = open_ipc_array(h_cluster, (frame_shape[0], frame_shape[1], 1), cp.uint8)
        d_roi = open_ipc_array(h_roi, (1, 4), cp.int32)
        
        predictor_ready.clear()
        ready_cluster_event.clear()
        roi_init = False
        cls_index = 0
        d_cluster[:] = 0
        
        dummy = np.zeros((384, 640, 3), dtype=np.uint8)
        result = model.predict(dummy, conf=0.1, verbose=False) 
        binary_mask = torch.zeros(size=(384, 640), dtype=torch.bool, device=model.device)
        predictor_ready.set()
   
        while not stop_event.is_set():
            if ready_frame_event.is_set():
                frame_cpu = cp.asnumpy(d_frame) 
                #t_frame = torch.as_tensor(d_frame, device=device).permute(2, 0, 1).unsqueeze(0).type(torch.float32) / 255
                #t_frame = F.interpolate(t_frame, size=(640, 640), mode='bilinear', align_corners=False)
                result = model.predict(frame_cpu, conf=0.1, verbose=False)[0]

                if ready_roi_event.is_set():
                    if result.boxes:
                        roi_local = cp.asnumpy(d_roi)
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

                d_cluster[:] = 0
                
                if not roi_init:
                    if result.masks is not None:
                        binary_mask[:] = False
                        for mask in result.masks.data:
                            binary_mask |= mask > 0.5
                        update_clusters(binary_mask, (result.masks.orig_shape[0], result.masks.orig_shape[1]))
                else:
                    if result.masks is not None :
                        binary_mask[:] = False
                        classes = result.boxes.cls.cpu().numpy().astype(np.int32)
                        for box_i, cls_i in enumerate(classes):
                            if cls_i != cls_index: continue
                            binary_mask |= result.masks.data[box_i, ::] > 0.5
                        update_clusters(binary_mask, (result.masks.orig_shape[0], result.masks.orig_shape[1]))
                
                cp.cuda.Stream.null.synchronize()
                ready_frame_event.clear()
                ready_cluster_event.set()

def launch_processes(server, args, device : str) -> None:
    cmr_clr_width, cmr_clr_height = producer_cli.map_to_camera_res[args.realsense_clr_stream]
    cmr_depth_width, cmr_depth_height = producer_cli.map_to_camera_res[args.realsense_depth_stream]
    cmr_fps = args.realsense_target_fps
    debug = args.debug

    stop_event = mp.Event()
    ready_frame_event = mp.Event()
    ready_cluster_event = mp.Event()
    ready_roi_event = mp.Event()
    predictor_event = mp.Event()
    
    ipc_queue = mp.Queue()

    cp.cuda.Device(0).use()
    d_shared_frame = cp.zeros((cmr_clr_height, cmr_clr_width, 3), dtype=cp.uint8)
    d_shared_cluster = cp.zeros((cmr_clr_height, cmr_clr_width, 1), dtype=cp.uint8)
    d_shared_roi = cp.zeros((1, 4), dtype=cp.int32)

    h_frame = cp.cuda.runtime.ipcGetMemHandle(d_shared_frame.data.ptr)
    h_cluster = cp.cuda.runtime.ipcGetMemHandle(d_shared_cluster.data.ptr)
    h_roi = cp.cuda.runtime.ipcGetMemHandle(d_shared_roi.data.ptr)
    ipc_queue.put((h_frame, h_cluster, h_roi))

    if args.cluster_predictor == 'sam2':
        enum = producer_cli.map_to_enum[args.sam2_checkpoint]
        link = producer_cli.map_to_config[enum]
        path_to_yaml = os.path.join(producer_cli.CONFIG_PATH, link[0])
        path_to_chkp = os.path.join(producer_cli.CHECKPOINT_PATH, Path(link[1]).name)

        if args.sam2_image_size % 32 != 0:
            args.sam2_image_size = 1024

        if not os.path.exists(path_to_chkp):
            os.makedirs(producer_cli.CHECKPOINT_PATH, exist_ok=True)
            producer_cli.getRequest(producer_cli.CHECKPOINT_PATH, link[1])
        
        predictor_proc = mp.Process(target=thread_worker_sam2, 
            args=(path_to_yaml, path_to_chkp, device, args.sam2_image_size, 
                  ipc_queue, 
                  (cmr_clr_height, cmr_clr_width), stop_event, ready_frame_event, ready_cluster_event, ready_roi_event, predictor_event))
                  
    elif args.cluster_predictor == 'yolo':
        if args.yolo_size == 'large':
            predictor = YOLOE("yoloe-11l-seg.pt", verbose=False)
        else:
            predictor = YOLOE("yoloe-11s-seg.pt", verbose=False)

        names = ["person"]
        predictor.set_classes(names, predictor.get_text_pe(names))

        predictor_proc = mp.Process(target=thread_worker_yoloe, 
            args=(predictor, device, 
                  ipc_queue, 
                  (cmr_clr_height, cmr_clr_width), stop_event, ready_frame_event, ready_cluster_event, ready_roi_event, predictor_event))
    else:
        print('Failed to parse predictor')
        return
    
    try:
        predictor_proc.start()

        print("Waiting for worker initialization...")
        while not predictor_event.is_set():
            time.sleep(0.1)
        print("Worker Ready!")

        camera_process(server, 
            d_shared_frame, d_shared_cluster, d_shared_roi,
            cmr_clr_width, cmr_clr_height,
            cmr_depth_width, cmr_depth_height, cmr_fps,
            stop_event, ready_frame_event, ready_cluster_event, ready_roi_event,
            args.point_cloud_budget,
            args.min_depth_meter,
            args.max_depth_meter,
            debug)
        
        predictor_proc.terminate()
        predictor_proc.join()
    except KeyboardInterrupt:
        print("Stopping processes...")
        stop_event.set()
        if predictor_proc.is_alive():
            predictor_proc.terminate()
        predictor_proc.join()