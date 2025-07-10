import time


import pyrealsense2 as rs
import cv2
import numpy as np

from encoding import *
import encoding as enc
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.live import Live
from rich.console import Group

import statslogger as log
from rich.table import Table
def main():

    # Setup RealSense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(cfg)


    prev_time = time.perf_counter()
    frame_count = 0
    fps = 0.0

    
    win_name = "RealSense Color"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    mode = Mode[Mode.FULL.name]

    statsROI = EncodingStats()
    statsOut = EncodingStats()
    statsAll = EncodingStats()
    statsGeneral = GeneralStats()

    executor = ProcessPoolExecutor(max_workers=2)
    with Live(refresh_per_second=10, screen=False) as live:
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
                color_frame = frames.get_color_frame() or frames.get_infrared_frame()
                depth_frame = frames.get_depth_frame()
                

                # Prepare image for display
                
                color_img = np.asanyarray(color_frame.get_data())        
                color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                depth_img = np.asanyarray(depth_frame.get_data())
                

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
                        
                h, w = depth_frame.get_height(), depth_frame.get_width()

                # Flatten and mask arrays

                u = (tex[:,0] * (w-1)).astype(np.int32)
                v = (tex[:,1] * (h-1)).astype(np.int32)
                # Clip to valid range
                u_idx = np.clip(u, 0, w-1)
                v_idx = np.clip(v, 0, h-1) # 3ms
                
                
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
                    cols_all = colors[valid] # 5ms
                    statsGeneral.prep_ms = (time.perf_counter() - prep_time_start) * 1000 #prep end

                    encode_time_start = time.perf_counter()
                    buf_all = dracoAll.encode(pts_all, cols_all, statsAll)
                    statsAll.encode_ms = (time.perf_counter() - encode_time_start) * 1000 # encoding end
                    statsAll.encoded_bytes = len(buf_all)

                    # Logging
                    tbl_general = log.make_general_stats_table(statsGeneral, "General Stats")
                    tbl_full = log.make_encoding_stats_table(statsAll, "Full-frame Stats")
                    total_time = statsAll.encode_ms + statsGeneral.get_total_time()
                    tbl_time = log.make_total_time_table(total_time)

                    live.update(Group(tbl_general, tbl_full, tbl_time))
                    
                else:
                    
                    yy, xx = np.divmod(np.arange(count), w) # 5 ms

                    # Draw ROI 
                    cx, cy = w // 2, h // 2
                    rw, rh = dracoROI.roiWidth, dracoROI.roiHeight
                    x0, y0 = max(0, cx - rw // 2), max(0, cy - rh // 2)
                    x1, y1 = min(w, x0 + rw), min(h, y0 + rh)
                    cv2.rectangle(color_bgr, (x0, y0), (x1, y1), (0,255,0), 2)

                    # Importance: bin ROI vs outside
                    in_roi = valid&(xx>=x0) & (xx<x1) & (yy>=y0) & (yy<y1)
                    out_roi = valid& ~in_roi
                    pts_roi = verts[in_roi]; cols_roi = colors[in_roi]
                    pts_out = verts[out_roi]; cols_out = colors[out_roi] # 6ms
                    statsGeneral.prep_ms = (time.perf_counter() - prep_time_start) * 1000

                    true_encoding_time = time.perf_counter()
                    f_roi = executor.submit(enc._encode_chunk, pts_roi, cols_roi, dracoROI, statsROI)
                    f_out = executor.submit(enc._encode_chunk, pts_out, cols_out, dracoOut, statsOut)

                    for fut in as_completed([f_roi, f_out]):
                        
                        buf, stats = fut.result()
                        if fut is f_roi:
                            buf_roi , statsROI = buf, stats
                        else:
                            buf_out, statsOut = buf, stats
                    true_encoding_time = (time.perf_counter() - true_encoding_time) * 1000

                    statsROI.encoded_bytes = len(buf_roi)
                    statsOut.encoded_bytes = len(buf_out)

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
                cv2.putText(color_bgr, f"FPS: {fps}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(color_bgr, dracoROI.to_string(), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                cv2.imshow(win_name, color_bgr)      
                key = cv2.waitKey(1) # this takes 20 ms
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
                elif key == ord(' '):
                    # save full point cloud PLY
                    points.export_to_ply("snapshot.ply", color_frame)
                    print("Saved full point cloud to snapshot.ply")

            
        finally:
            pipeline.stop()

if __name__ == "__main__":
    main()