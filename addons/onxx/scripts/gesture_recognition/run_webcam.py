import time

import cv2

from mediapipe.tasks.python.vision import (
    RunningMode,
)

from finger_detection import (
    FingerDirection,
    PixelBoundingBox,
)


finger_direction_recognizer = FingerDirection(
    model_asset_path="hand_landmarker.task",
    num_hands=2,
    running_mode=RunningMode.LIVE_STREAM,
    box_size=0.2,
    delay_frames=15,
)

cap = cv2.VideoCapture(6)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

fps_reported = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera reports FPS = {fps_reported:.2f}")

frame_count = 0
start_time  = time.time()

while True:
    # 2. Read the latest frame from the camera
    success, numpy_frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break

    rgb_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_BGR2RGB)

    mediapipe_image = FingerDirection.convert_frame(rgb_frame=rgb_frame)
    timestamp_ms = int(time.time() * 1000)

    finger_direction_recognizer.recognize(mediapipe_image, frame_timestamp_ms=timestamp_ms)

    for bounding_box_normalized in finger_direction_recognizer.latest_bounding_boxes:
        if bounding_box_normalized:
            pixel_space_bounding_box: PixelBoundingBox = bounding_box_normalized.to_pixel(numpy_frame.shape[1], numpy_frame.shape[0])
            cv2.rectangle(numpy_frame, (pixel_space_bounding_box.x1, pixel_space_bounding_box.y1), (pixel_space_bounding_box.x2, pixel_space_bounding_box.y2), (0,255,0), 2)
    
    height, width = numpy_frame.shape[:2]
    masks = finger_direction_recognizer.get_masks((height, width))

    for i in range(masks.shape[0]):
        # each mask is 0/1; multiply by 255 to make it visible
        cv2.imshow(f"finger-mask-{i}", masks[i] * 255)

    cv2.imshow("Webcam", numpy_frame)

    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        print(f"Measured FPS = {frame_count/elapsed:.2f}")
        frame_count = 0
        start_time  = time.time()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


