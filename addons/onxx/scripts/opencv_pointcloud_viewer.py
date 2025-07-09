# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

import math
import time
import cv2
import torch
import sam2_config
import argparse
import os

from pathlib import Path

import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import sam2_camera_predictor as sam2_camera

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(0), math.radians(0)
        self.translation = np.array([0, 0, 0], dtype=np.float32)
        self.distance = 0
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = False
        self.color = True
        self.apply_hole_filter = True
        self.apply_decimate = False

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

def launch_demo(path_to_yaml: str,  path_to_chkp: str, device: str, image_size: int):
    config_name = Path(path_to_yaml).name
    config_path = "configs"

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    predictor = sam2_camera.build_sam2_camera_predictor(
        config_file=config_name, 
        config_path=config_path,
        ckpt_path=path_to_chkp, 
        device=device,
        image_size=image_size
    )

    state = AppState()

    point = None
    action_now = False
    if_init = False

    def mouse_cb(event, x, y, flags, param):
        nonlocal point, action_now, if_init
        if event == cv2.EVENT_LBUTTONDOWN:
            state.mouse_btns[0] = True
            point = [x, y]
            action_now = True
            if_init = False

        if event == cv2.EVENT_LBUTTONUP:
            state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = out.shape[:2]
            dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

            if state.mouse_btns[0]:
                state.yaw += float(dx) / w * 2
                state.pitch -= float(dy) / h * 2

            elif state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                state.translation -= np.dot(state.rotation, dp)

            elif state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                state.translation[2] += dz
                state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            state.translation[2] += dz
            state.distance -= dz

        state.prev_mouse = (x, y)

    def project(v):
        """project 3d vector array to 2d"""
        h, w = out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj

    def view(v):
        """apply view transformation on vector array"""
        return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

    def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = project(pt1.reshape(-1, 3))[0]
        p1 = project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

    def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
                view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
                view(pos + np.dot((s2, 0, z), rotation)), color)

    def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        line3d(out, pos, pos +
            np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        line3d(out, pos, pos +
            np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        line3d(out, pos, pos +
            np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)

    def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                line3d(out, orig, view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            line3d(out, view(top_left), view(top_right), color)
            line3d(out, view(top_right), view(bottom_right), color)
            line3d(out, view(bottom_right), view(bottom_left), color)
            line3d(out, view(bottom_left), view(top_left), color)

    def pointcloud(out, verts, texcoords, color, painter=True, freeze=False):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            if not freeze:
                v = view(verts)
            else:
                v = verts - np.array([0, 0, 0], dtype=np.float32)

            s = v[:, 2].argsort()[::-1]
            proj = project(v[s])
        else:
            proj = project(view(verts))

        if state.scale:
            proj *= 0.5**state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        uv = texcoords[s] * (cw - 1, ch - 1)
        uv = np.clip(uv, 0, [cw - 1, ch - 1])
        u, v = uv.astype(np.uint32).T

        # Use v = row (y), u = col (x)
        out[i[m], j[m]] = color[v[m], u[m]]

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    thresh_filter = rs.threshold_filter()
    thresh_filter.set_option(rs.option.min_distance, 0.01)
    thresh_filter.set_option(rs.option.max_distance, 1.5)
    hole_filter = rs.hole_filling_filter()
    hole_filter.set_option(rs.option.holes_fill, 1)
    decimate.set_option(rs.option.filter_magnitude, 1)
    colorizer = rs.colorizer()

    spatial = rs.spatial_filter()

    spatial.set_option(rs.option.filter_magnitude, 2)  # Smoothing extent (default 2, range 1–5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # Alpha: smoothing across frames
    spatial.set_option(rs.option.filter_smooth_delta, 10)   # Delta: depth discontinuity tolerance
    spatial.set_option(rs.option.holes_fill, 5)  # 0–5 (aggressiveness of hole filling)

    temporal = rs.temporal_filter()
    # Optional: adjust smoothing parameters
    temporal.set_option(rs.option.filter_smooth_alpha, 1)  # [0-1] closer to 1 = smoother
    temporal.set_option(rs.option.filter_smooth_delta, 10)   # Max allowed depth change
    temporal.set_option(rs.option.holes_fill, 5)             # 0–5: fills holes in new frame from previous

    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(state.WIN_NAME, w, h)
    cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

    out = np.empty((h, w, 3), dtype=np.uint8)
    out_front = np.empty((h, w, 3), dtype=np.uint8)

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        if not state.paused:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if state.apply_decimate:
                depth_frame = decimate.process(depth_frame)

            #if state.apply_hole_filter:
            #    depth_frame = hole_filter.process(depth_frame)

           # depth_frame = thresh_filter.process(depth_frame)
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            if state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

            out_mask_logits = None

            if action_now:
                predictor.load_first_frame(color_image)

                ann_frame_idx = 0
                ann_obj_id = (1,)
                labels = np.array([1], dtype=np.int32)
                points = np.array([point], dtype=np.float32)

                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
                )

                if_init = True
                action_now = False
            else:
                if if_init:
                    out_obj_ids, out_mask_logits = predictor.track(color_image)

        results = hands.process(color_image)
        tmp_img = color_image.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        now = time.time()

        out.fill(0)
        out_front.fill(0)

        grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            v = verts.copy()
            t = texcoords.copy()
            pointcloud(out, verts, texcoords, tmp_img)
            pointcloud(out_front, v, t, tmp_img, True, True)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, tmp_img)
            tmp = cv2.resize(tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now

        cv2.setWindowTitle(
            state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            (w, h, 1.0/(dt + 1.e-5), dt*1000, "PAUSED" if state.paused else ""))

        if out_mask_logits != None:
            all_mask = np.zeros_like(color_image, dtype=np.uint8)
                
            out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            colored_mask = np.zeros_like(color_image, dtype=np.uint8)

            colored_mask[:, :, 0] = out_mask[:, :, 0] * 255
            colored_mask[:, :, 1] = out_mask[:, :, 0] * 0
            colored_mask[:, :, 2] = out_mask[:, :, 0] * 0

            all_mask = cv2.addWeighted(all_mask, 0, colored_mask, 1, 0)
            image_seg = cv2.addWeighted(tmp_img, 1, all_mask, 1, 0)

            final_out = np.hstack((color_image, image_seg, depth_colormap, out_front, out))
        else:
            final_out = np.hstack((color_image, depth_colormap, out_front, out))

        cv2.imshow(state.WIN_NAME, final_out)
        key = cv2.waitKey(1)

        if key == ord("r"):
            state.reset()

        if key == ord("p"):
            state.paused ^= True

        if key == ord("d"):
            state.decimate = (state.decimate + 1) % 3
            decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

        if key == ord("z"):
            state.scale ^= True

        if key == ord("c"):
            state.color ^= True

        if key == ord("s"):
            cv2.imwrite('./out.png', final_out)

        if key == ord("e"):
            points.export_to_ply('./out.ply', mapped_frame)

        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            break

    pipeline.stop()


def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Realsense SAM2.1 demo")
    parser.add_argument("--checkpoint", type=str, default="large", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--image_size", type=int, default=1024)

    args = parser.parse_args()

    enum = sam2_config.map_to_enum[args.checkpoint]
    link = sam2_config.map_to_config[enum]
    path_to_yaml = os.path.join(sam2_config.CONFIG_PATH, link[0])
    print(f"Received path_to_yaml: {path_to_yaml}")
    path_to_chkp = os.path.join(sam2_config.CHECKPOINT_PATH, Path(link[1]).name)

    if args.image_size % 32 != 0:
        print(f'Requested image size {args.image_size} is not a multple of 32 falling back to SAM2.1 default 1024')
        args.image_size = 1024

    if not os.path.exists(sam2_config.CONFIG_PATH):
        print('Config path for sam2.1 does not exist, exiting...')
        return
    
    if not os.path.exists(path_to_yaml):
        print(f'Config {link[0]} for sam2.1 does not exist, You need to download them from https://github.com/facebookresearch/sam2/tree/main/sam2/configs/sam2.1, exiting...')
        return
    
    if not os.path.exists(path_to_chkp):
        print(f'Checkpoint {path_to_chkp} is missing, downloading...')
        os.makedirs(sam2_config.CHECKPOINT_PATH, exist_ok=True)
        sam2_config.getRequest(sam2_config.CHECKPOINT_PATH, link[1])

    with torch.autocast(device_type=DEVICE.__str__(), dtype=torch.bfloat16):
        launch_demo(
            path_to_yaml=path_to_yaml, 
            path_to_chkp=path_to_chkp, 
            device=DEVICE,
            image_size=args.image_size
        )

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Compiled CUDA version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    main()