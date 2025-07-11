import os
import numpy as np
import cv2 as cv
import time
from ultralytics import YOLOE

CWD = os.getcwd()

model = YOLOE("yoloe-11l-seg.pt", verbose=False)
names = ["glasses", "shirt", "hat", "shorts", "wedding ring"]
model.set_classes(names, model.get_text_pe(names))

cap = cv.VideoCapture(0)
cv.namedWindow("Camera")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    now = time.time()

    results = model.predict(frame, conf=0.1, verbose=False)

    #results[0].show()

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.int32)
        cls = r.boxes.cls.cpu().numpy()
        #probs = r.probs
        #masks = r.masks.data.cpu().permute(1, 2, 0).numpy() >= 0.0
        #color_mask = np.zeros_like(frame)
        #color_mask[masks[:, :, 0], :] = np.random.rand(3)
        for (box, c) in zip(boxes, cls):
            cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            cv.putText(frame, model.names[c], (box[0] - 1, box[1] - 1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        #frame = cv.addWeighted(frame, 0.5, color_mask, 0.5, 0)

    dt = time.time() - now


    cv.setWindowTitle("Camera", f"{1.0/(dt + 1.e-5):.2f}FPS ({dt*1000:.2f}ms)")
    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()

