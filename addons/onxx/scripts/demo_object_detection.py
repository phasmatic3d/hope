import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import cv2 as cv
import time

CWD = os.getcwd()
CHECKPOINT = 'custom_model.tflite'
CHECKPOINT_PATH = os.path.join(CWD, 'checkpoints')

interpreter = tf.lite.Interpreter(model_path=os.path.join(CHECKPOINT_PATH, CHECKPOINT))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv.VideoCapture(0)
cv.namedWindow("Camera")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    now = time.time()
    input_shape = input_details[0]['shape']
    image_resized = cv.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    print(len(output_details))
    
    boxes = interpreter.get_tensor(output_details[0]['index'])
    dt = time.time() - now
    #classes = interpreter.get_tensor(output_details[1]['index'])
    #scores = interpreter.get_tensor(output_details[2]['index'])

    #print("Detections:", boxes, classes, scores)

    cv.setWindowTitle("Camera", f"{1.0/(dt + 1.e-5):.2}FPS ({dt*1000:.2}ms)")
    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()

