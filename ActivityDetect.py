import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the class indices for "person" and "cell phone"
person_class = 0  # 'person' class ID
mobile_class = 67  # 'cell phone' class ID for YOLOv5 COCO dataset

# Load a sample video
video = cv2.VideoCapture(r"C:\Users\USER\ActivityDetection\SampleAcitvityDetection.mp4")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break


video = cv2.VideoCapture("SampleAcitvityDetection.mp4")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Run the YOLOv5 model on the frame
    results = model(frame)

    # Convert predictions into a pandas DataFrame
    predictions = results.pandas().xyxy[0]

    # Filter predictions for 'person' and 'cell phone'
    people = predictions[predictions['class'] == person_class]
    phones = predictions[predictions['class'] == mobile_class]

    # Draw bounding boxes for each person and phone
    for _, row in people.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Person', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    for _, row in phones.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Mobile', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Phone Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
