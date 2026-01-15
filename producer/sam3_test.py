import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from sam3_camera_predictor import build_sam3_camera_predictor

# --- 1. RealSense Setup ---
def initialize_realsense(width=1280, height=720, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the Color stream (SAM 3 works on RGB images)
    # We use BGR8 format to match OpenCV's default color space
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    print(f"Starting RealSense Pipeline ({width}x{height} @ {fps}fps)...")
    profile = pipeline.start(config)
    return pipeline, profile

# --- 2. SAM 3 Setup ---
checkpoint_path = "checkpoints/sam3_hiera_l.pt"  # Adjust path if needed
config_path = "sam3/configs/sam3_hiera_l.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SAM 3 on {device}...")

predictor = build_sam3_camera_predictor(
    config_file="sam3_hiera_l.yaml",
    ckpt_path=checkpoint_path,
    device=device
)

# --- 3. Main Loop ---
def main():
    pipeline, profile = initialize_realsense()
    
    try:
        # Warmup: Wait for a valid frame
        for _ in range(5):
            pipeline.wait_for_frames()
            
        # Get first valid frame for initialization
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Could not get first frame from RealSense")
            
        # Convert to numpy array (H, W, 3)
        initial_frame = np.asanyarray(color_frame.get_data())
        
        # Initialize SAM 3 with the first frame
        predictor.load_first_frame(initial_frame)
        
        obj_id = 1
        print("\n--- Controls ---")
        print("  't': Track new object (type text prompt)")
        print("  'c': Clear/Reset tracking")
        print("  'q': Quit")
        print("----------------")

        while True:
            # A. Get Frame from RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert to standard BGR image for OpenCV/SAM
            frame = np.asanyarray(color_frame.get_data())

            # B. Add frame to SAM 3 buffer
            predictor.add_conditioning_frame(frame)

            # C. Run Tracking
            out_obj_ids, out_masks = predictor.track(frame)

            # D. Visualization
            vis_frame = frame.copy()

            for i, out_obj_id in enumerate(out_obj_ids):
                # SAM 3 returns masks on GPU; move to CPU for drawing
                mask = out_masks[i, 0].cpu().numpy()
                mask_binary = (mask > 0.0).astype(np.uint8)

                if mask_binary.sum() > 0:
                    # Color Overlay (Green)
                    color = (0, 255, 0)
                    colored_mask = np.zeros_like(vis_frame)
                    colored_mask[mask_binary == 1] = color
                    vis_frame = cv2.addWeighted(vis_frame, 1.0, colored_mask, 0.5, 0)

                    # Bounding Box & Label
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Find largest contour (main object)
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(vis_frame, f"ID {out_obj_id}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("SAM 3 + RealSense", vis_frame)
            key = cv2.waitKey(1) & 0xFF

            # E. User Interaction
            if key == ord('t'):
                text = input("Enter object to track (e.g., 'cup', 'headphones'): ")
                current_frame_idx = predictor.condition_state["num_frames"] - 1
                
                print(f"Initializing tracking for '{text}'...")
                predictor.add_text_prompt(
                    frame_idx=current_frame_idx, 
                    obj_id=obj_id, 
                    text_prompt=text
                )
                obj_id += 1 # Increment ID for next object

            elif key == ord('c'):
                print("Resetting state...")
                predictor.reset_state()
                predictor.load_first_frame(frame)
                obj_id = 1

            elif key == ord('q'):
                break

    finally:
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()