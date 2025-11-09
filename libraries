from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import time
import numpy as np
model = YOLO("yolov8n.pt")  # you can use yolov8s.pt for better accuracy
print("✅ YOLOv8 model loaded successfully!")
if not os.path.exists("violations"):
    os.makedirs("violations")
    print("📂 'violations' folder created.")
else:
    print("📁 'violations' folder already exists.")
# Open webcam (0) or use a video file path
cap = cv2.VideoCapture(0)  # replace 0 with 'traffic.mp4' if you have a video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    person_count = 0
    helmet_detected = False
    seatbelt_detected = False

    # loop through detections
    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]

        if label == "person":
            person_count += 1
        elif label == "helmet":
            helmet_detected = True
        elif label == "seatbelt":
            seatbelt_detected = True

    alert_text = ""

    if person_count > 2:
        alert_text += "⚠️ Vehicle Overloaded | "
    if not helmet_detected:
        alert_text += "⚠️ No Helmet | "
    if not seatbelt_detected:
        alert_text += "⚠️ No Seatbelt | "

    if alert_text:
        cv2.putText(annotated_frame, alert_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        filename = f"violations/violation_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)

    cv2.imshow("Vehicle Safety Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
