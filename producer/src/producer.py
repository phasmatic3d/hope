import argparse
import time
import threading
from enum import Enum, auto

import pyrealsense2 as rs
import cv2
import numpy as np
import DracoPy

# Modes for processing
class Mode(Enum):
    FULL = auto()          
    IMPORTANCE = auto()    


class Timer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.start = time.time()
    def elapsed_ms(self):
        return (time.time() - self.start) * 1000


class EncodingStats:
    def __init__(self):
        self.pc_ms = 0
        self.prep_ms = 0
        self.encode_ms = 0
        self.pts = 0
        self.raw_bytes = 0
        self.encoded_bytes = 0
    @property
    def total_time_ms(self):
        return self.pc_ms + self.prep_ms + self.encode_ms

class DracoSettings:
    def __init__(self):
        self.posQuant = 10
        self.colorQuant = 8
        self.speedEncode = 1
        self.speedDecode = 1
        self.roiWidth = 240
        self.roiHeight = 240
    def to_string(self):
        return f"Qpos:{self.posQuant:2d} Qcol:{self.colorQuant:2d} Spd:{self.speedEncode}/10 ROI:{self.roiWidth}x{self.roiHeight}"

    def encode(self, points, colors, stats: EncodingStats):

        stats.pts = points.shape[0]
        stats.raw_bytes = stats.pts * (3 * 4 + 3 * 1)
        t_prep = Timer()

        stats.prep_ms = t_prep.elapsed_ms()

        t_enc = Timer()

        buffer = DracoPy.encode(
            points,
            quantization_bits=self.posQuant,
            compression_level=self.speedEncode,
            colors=colors
        )
        stats.encode_ms = t_enc.elapsed_ms()
        stats.encoded_bytes = len(buffer)
        return buffer

# Main demo

def main():

    # Setup RealSense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(cfg)


    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    
    win_name = "RealSense Color"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    mode = Mode[Mode.FULL.name]

    statsROI = EncodingStats()
    statsOut = EncodingStats()

    try:
        while True:
            # Settings
            dracoROI = DracoSettings()

            dracoOut = DracoSettings()
            dracoOut.posQuant = dracoROI.posQuant // 4
            dracoOut.colorQuant = dracoROI.colorQuant
            dracoOut.speedEncode = dracoROI.speedEncode
            dracoOut.speedDecode = dracoROI.speedDecode



            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame() or frames.get_infrared_frame()
            depth_frame = frames.get_depth_frame()

            # Prepare image for display
            color_img = np.asanyarray(color_frame.get_data())
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            # Depth image for hole detection
            depth_img = np.asanyarray(depth_frame.get_data())

            # Generate point cloud
            rpc = rs.pointcloud()
            rpc.map_to(color_frame)
            points = rpc.calculate(depth_frame)

            count = points.size()
            vert_buf = points.get_vertices()
            tex_buf = points.get_texture_coordinates()

            verts = np.frombuffer(vert_buf, dtype=np.float32).reshape(count, 3)
            tex = np.frombuffer(tex_buf, dtype=np.float32).reshape(count, 2)

            h, w = depth_frame.get_height(), depth_frame.get_width()


            # Flatten and mask arrays
            depth_flat = depth_img.ravel()
            u_idx = np.clip((tex[:,0]*(w-1)).round().astype(int), 0, w-1)
            v_idx = np.clip((tex[:,1]*(h-1)).round().astype(int), 0, h-1)
            pix_idx = v_idx * w + u_idx
            bgr_flat = color_bgr.reshape(-1,3)
            colors = bgr_flat[pix_idx][:, ::-1]
            yy, xx = np.divmod(np.arange(count), w)
            valid = (depth_flat>0) & np.isfinite(verts[:,2]) & (verts[:,2]>0)

            buf_all = None
            buf_out = None
            buf_roi = None
            if mode is Mode.FULL:
                # Encode entire valid cloud
                pts_all = verts[valid]
                cols_all = colors[valid]

                
                statsAll = EncodingStats()
                print("====Encoding Stats (DEBUG)====")
                encode_now = time.time()
                buf_all = dracoROI.encode(pts_all, cols_all, statsAll)
                end_encode = time.time()
                print("Python-side Encode()", (end_encode - encode_now) * 1000, "ms")
                print()
            else:
                # Draw ROI rectangle on color image
                cx, cy = w // 2, h // 2
                rw, rh = dracoROI.roiWidth, dracoROI.roiHeight
                x0, y0 = max(0, cx - rw // 2), max(0, cy - rh // 2)
                x1, y1 = min(w, x0 + rw), min(h, y0 + rh)
                cv2.rectangle(color_bgr, (x0, y0), (x1, y1), (0,255,0), 2)

                # Importance: bin ROI vs outside
                in_roi = valid&(xx>=x0) & (xx<x1) & (yy>=y0) & (yy<y1)
                out_roi = valid& ~in_roi
                pts_roi = verts[in_roi]; cols_roi = colors[in_roi]
                pts_out = verts[out_roi]; cols_out = colors[out_roi]
                
                
                statsROI = EncodingStats()
                print("====Encoding Stats (DEBUG)====")
                encode_now = time.time()
                buf_roi = dracoROI.encode(pts_roi, cols_roi, statsROI)
                
                statsOut = EncodingStats()
                buf_out = dracoOut.encode(pts_out, cols_out, statsOut)
                end_encode = time.time()
                print("Python-side Encode()", (end_encode - encode_now) * 1000, "ms")
                print()


            
            # Display FPS
            frame_count += 1
            now = time.time()
            fps = frame_count / (now - prev_time)
            prev_time = now; frame_count = 0
            cv2.putText(color_bgr, f"FPS: {fps:.1f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(color_bgr, dracoROI.to_string(), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow(win_name, color_bgr)
            key = cv2.waitKey(1)
            if key in (ord('q'), 27):  # q or Esc
                break
            # Adjust settings
            elif key == ord('f'):
                # Toggle mode
                mode = Mode.FULL if mode is Mode.IMPORTANCE else Mode.IMPORTANCE
            elif key == ord('='):
                dracoROI.posQuant = min(dracoROI.posQuant+1, 20)
            elif key == ord('-'):
                dracoROI.posQuant = max(dracoROI.posQuant-1, 1)
            elif key == ord(']'):
                dracoROI.colorQuant = min(dracoROI.colorQuant+1, 16)
            elif key == ord('['):
                dracoROI.colorQuant = max(dracoROI.colorQuant-1, 1)
            elif key == ord('f'):
                # toggle full vs ROI mode (not implemented)
                pass
            elif key == ord(' '):
                # save full point cloud PLY
                points.export_to_ply("snapshot.ply", color_frame)
                print("Saved full point cloud to snapshot.ply")

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()