from enum import Enum, auto
import time
import numpy as np

import encoder
import encoding as enc


from typing import Tuple
from broadcasting import *

import pyrealsense2 as rs
import cv2

from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.live import Live
from rich.console import Group

import statslogger as log


# Modes for processing
class Mode(Enum):
    FULL = auto()          
    IMPORTANCE = auto()    

class VizMode(Enum):
    COLOR = auto()
    DEPTH = auto()


class Timer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.start = time.time()
    def elapsed_ms(self):
        return (time.time() - self.start) * 1000


class DracoEncoder:
    def __init__(self):
        self.posQuant = 10
        self.colorQuant = 8
        self.speedEncode = 9
        self.speedDecode = 9
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

def _encode_chunk(pts: np.ndarray,
                colors: np.ndarray,
                encoder: DracoEncoder,
                stats: log.EncodingStats
                ) -> Tuple[bytes, log.EncodingStats]:
# instantiate a fresh encoder in this process
    start = time.time()
    buf = encoder.encode(pts, colors, stats)
    end = time.time()
    stats.encode_ms = (end - start) * 1000
    return buf, stats

def encode_point_cloud(server: bc.ProducerServer):
    # Setup RealSense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    pipeline.start(cfg)

    align_to = rs.stream.color
    align     = rs.align(align_to)



    prev_time = time.perf_counter()
    frame_count = 0
    fps = 0.0

    
    win_name = "RealSense Color"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    mode = Mode[Mode.FULL.name]

    statsROI = log.EncodingStats()
    statsOut = log.EncodingStats()
    statsAll = log.EncodingStats()
    statsGeneral = log.GeneralStats()

    executor = ProcessPoolExecutor(max_workers=2)

    min_dist = 0.1
    max_dist = 2.0
    depth_thresh = rs.threshold_filter(min_dist, max_dist)
    viz_mode = VizMode.COLOR

    with Live(refresh_per_second=1, screen=False) as live:
        try:
            while True:
                # Settings
                dracoAll = DracoEncoder()
                dracoROI = DracoEncoder()

                dracoOut = DracoEncoder()
                dracoOut.posQuant = dracoROI.posQuant // 4
                dracoOut.colorQuant = dracoROI.colorQuant
                dracoOut.speedEncode = dracoROI.speedEncode
                dracoOut.speedDecode = dracoROI.speedDecode

                
                frames = pipeline.wait_for_frames()

                statsGeneral.frame_ms = time.perf_counter()
                frames = align.process(frames)
                statsGeneral.frame_ms = (time.perf_counter() - statsGeneral.frame_ms) *1000

                depth_frame = frames.get_depth_frame()  
                color_frame = frames.get_color_frame()    
                          
                h, w = depth_frame.get_height(), depth_frame.get_width()

                statsGeneral.cull_ms = time.perf_counter()
                depth_frame = depth_thresh.process(depth_frame)
                statsGeneral.cull_ms = (time.perf_counter() - statsGeneral.cull_ms) *1000

                

                # Prepare image for display
                
                color_img = np.asanyarray(color_frame.get_data())        
                color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                depth_img = np.asanyarray(depth_frame.get_data())
                

                # pick which to show
                if viz_mode is VizMode.COLOR:
                    display = color_bgr
                else:
                    depth_8u = cv2.convertScaleAbs(depth_img, alpha=255.0 / depth_img.max())
                    depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                    display = depth_colormap

                

                # Generate point cloud
                pc_time_start = time.perf_counter()
                rpc = rs.pointcloud()
                rpc.map_to(color_frame)
                points = rpc.calculate(depth_frame) # 6ms
                statsGeneral.pc_ms = (time.perf_counter() - pc_time_start) * 1000
                
                prep_time_start = time.perf_counter()
                count = points.size()
                vert_buf = points.get_vertices()
                tex_buf = points.get_texture_coordinates()
                
                verts = np.frombuffer(vert_buf, dtype=np.float32).reshape(count, 3)
                tex = np.frombuffer(tex_buf, dtype=np.float32).reshape(count, 2) # 1ms
                
                # Flatten and mask arrays
                
                u = (tex[:,0] * (w-1)).astype(np.int32)
                v = (tex[:,1] * (h-1)).astype(np.int32) # 3 ms

                # Clip to valid range
                u_idx = np.clip(u, 0, w-1)
                v_idx = np.clip(v, 0, h-1) # 1ms

                
                
                pix_idx = v_idx * w + u_idx
                bgr_flat = color_bgr.reshape(-1,3) 
                colors = bgr_flat[pix_idx][:, ::-1]# 4ms

                # find valid points
                depth_flat = depth_img.ravel()
                valid = (depth_flat>0) & np.isfinite(verts[:,2]) & (verts[:,2]>0) # 1ms
                
                

                buf_all = None
                buf_out = None
                buf_roi = None
                if mode is Mode.FULL:
                    # Encode entire valid cloud
                    pts_all = verts[valid]
                    cols_all = colors[valid] # 8ms
                    statsGeneral.prep_ms = (time.perf_counter() - prep_time_start) * 1000 #prep end

                    encode_time_start = time.perf_counter()
                    if(pts_all.size != 0):
                        buf_all = dracoAll.encode(pts_all, cols_all, statsAll)
                        statsAll.encode_ms = (time.perf_counter() - encode_time_start) * 1000 # encoding end
                        statsAll.encoded_bytes = len(buf_all)
                        statsGeneral.true_enc_ms  = statsAll.encode_ms

                        # Broadcast
                        server.broadcast(bytes([1]) + buf_all) # prefix with single byte to understand that we are sending one buffer
                    
                    # Logging
                    tbl_general = log.make_general_stats_table(statsGeneral, "General Stats")
                    tbl_full = log.make_encoding_stats_table(statsAll, "Full-frame Stats")
                    total_time = statsAll.encode_ms + statsGeneral.get_total_time()
                    tbl_time = log.make_total_time_table(total_time)

                    live.update(Group(tbl_general, tbl_full, tbl_time))
                    
                else:
                    
                    #TODO: REPLACE WITH SAM/GEST DETECTION
                    yy, xx = np.divmod(np.arange(count), w) # 6 ms
                    
                    # Draw ROI 
                    cx, cy = w // 2, h // 2
                    rw, rh = dracoROI.roiWidth, dracoROI.roiHeight
                    x0, y0 = max(0, cx - rw // 2), max(0, cy - rh // 2)
                    x1, y1 = min(w, x0 + rw), min(h, y0 + rh)
                    cv2.rectangle(display, (x0, y0), (x1, y1), (0,255,0), 2)

                    # Importance: bin ROI vs outside
                    
 
                    in_roi = valid&(xx>=x0) & (xx<x1) & (yy>=y0) & (yy<y1)
                    out_roi = valid& ~in_roi
                    pts_roi = verts[in_roi]; cols_roi = colors[in_roi]
                    pts_out = verts[out_roi]; cols_out = colors[out_roi] # 10ms
                    
                    statsGeneral.prep_ms = (time.perf_counter() - prep_time_start) * 1000
                    

                    # MULTIPROCESSING IMPORTANCE
                    true_encoding_time = time.perf_counter()
                    futures = []
                    if pts_roi.size: # check if non empty
                        futures.append(("roi",
                            executor.submit(enc._encode_chunk,
                                            pts_roi, cols_roi,
                                            dracoROI, statsROI)))
                    else:
                        buf_roi = b""

                    if pts_out.size:
                        futures.append(("out",
                            executor.submit(enc._encode_chunk,
                                            pts_out, cols_out,
                                            dracoOut, statsOut)))
                    else:
                        buf_out = b""


                    for name, fut in futures:
                        
                        buf, stats = fut.result()
                        if name == "roi":
                            buf_roi , statsROI = buf, stats
                        elif name == "out":
                            buf_out, statsOut = buf, stats
                    true_encoding_time = (time.perf_counter() - true_encoding_time) * 1000
                    statsGeneral.true_enc_ms = true_encoding_time
                    statsROI.encoded_bytes = len(buf_roi)
                    statsOut.encoded_bytes = len(buf_out)

                    pc_count = 0
                    # Broadcast
                    bufs = []
                    if buf_roi:
                        bufs.append(buf_roi)
                    if buf_out:
                        bufs.append(buf_out)
                    count = len(bufs)
                    for buf in bufs:
                        server.broadcast(bytes([count]) + buf) # Prefix with byte that tells us the length



                    # Logging
                    tbl_general = log.make_general_stats_table(statsGeneral, "General Stats")
                    tbl_roi = log.make_encoding_stats_table(statsROI, "In-Roi Stats")
                    tbl_out = log.make_encoding_stats_table(statsOut, "Out-of-ROI Stats")
                    total_time = true_encoding_time + statsGeneral.get_total_time()
                    tbl_time = log.make_total_time_table(total_time)

                    live.update(Group(tbl_general, tbl_roi,tbl_out,tbl_time))
                
            
                frame_count += 1
                now = time.perf_counter()
                if (frame_count >= 30):
                    fps = frame_count // (now - prev_time)
                    prev_time = now
                    frame_count = 0
                cv2.putText(display, f"FPS: {fps}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(display, dracoROI.to_string(), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                cv2.imshow(win_name, display)      
                key = cv2.waitKey(1) # this takes 20 ms
                if key in (ord('q'), 27):  # q or Esc
                    break
                # Adjust settings
                elif key == ord('f'):
                    # Toggle mode (IMPORTANCE)
                    mode = Mode.FULL if mode is Mode.IMPORTANCE else Mode.IMPORTANCE
                elif key == ord('d'):
                    #Toggle mode (Visualization: Depth vs Color)
                    viz_mode = (
                        VizMode.DEPTH
                        if viz_mode is VizMode.COLOR
                        else VizMode.COLOR
                    )
                elif key == ord('='):
                    dracoROI.posQuant = min(dracoROI.posQuant+1, 20)
                elif key == ord('-'):
                    dracoROI.posQuant = max(dracoROI.posQuant-1, 1)
                elif key == ord(']'):
                    dracoROI.colorQuant = min(dracoROI.colorQuant+1, 16)
                elif key == ord('['):
                    dracoROI.colorQuant = max(dracoROI.colorQuant-1, 1)
                elif key == ord(' '):
                    # save full point cloud PLY
                    points.export_to_ply("snapshot.ply", color_frame)
                    print("Saved full point cloud to snapshot.ply")

            
        finally:
            pipeline.stop()
