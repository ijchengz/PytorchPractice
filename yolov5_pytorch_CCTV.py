import torch
import numpy as np
import cv2
import time

prev_time = 0
url = 'https://thbcctv01.thb.gov.tw/T1-13K+600'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cap = cv2.VideoCapture(url)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.resize(image, (0, 0), fx=2.5, fy=2.5)
    results = model(image)
    output_image = np.squeeze(results.render())
    cv2.putText(output_image, f'FPS: {int(1 / (time.time() - prev_time))}', (3, 40),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    prev_time = time.time()
    cv2.imshow('pyTorch', output_image)
    results.print()
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()