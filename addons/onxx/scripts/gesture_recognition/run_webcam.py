import time

import cv2

from gesture_recognition import GestureRecognition


recognizer = GestureRecognition(model_asset_path="gesture_recognizer.task")

cap = cv2.VideoCapture(0)
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

    mediapipe_image = GestureRecognition.convert_frame(rgb_frame=rgb_frame)
    timestamp_ms = int(time.time() * 1000)


    recognizer.recognize(mediapipe_image, frame_timestamp_ms=timestamp_ms)

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


