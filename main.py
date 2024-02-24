from ultralytics import YOLO
import cv2
from networktables import NetworkTables
import json

import logging
logging.basicConfig(level=logging.DEBUG)

# Load the YOLOv8 model
model = YOLO('weights/best.pt')

cap = cv2.VideoCapture(0)

NetworkTables.initialize()
detector = NetworkTables.getTable("Detector")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()

        detector.putNumber("DetectedObjectsCount", len(boxes))

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            confidence = conf
            detected_class = cls
            name = names[int(cls)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            detector.putStringArray("Entries", [str(x1), str(y1), str(x2), str(y2), str(confidence)])

            cv2.putText(frame, f"{name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()


# # Extract bounding boxes, classes, names, and confidences
# boxes = results[0].boxes.xyxy.tolist()
# classes = results[0].boxes.cls.tolist()
# names = results[0].names
# confidences = results[0].boxes.conf.tolist()
#
# # Iterate through the results
# for box, cls, conf in zip(boxes, classes, confidences):
#     x1, y1, x2, y2 = box
#     confidence = conf
#     detected_class = cls
#     name = names[int(cls)]