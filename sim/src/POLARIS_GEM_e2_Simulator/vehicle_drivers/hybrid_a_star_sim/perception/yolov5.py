import numpy as np
import cv2
import torch

model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.conf = 0.25
    
def infer(img_path):
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        cv2.imshow('Original Image', frame)
        cv2.waitKey(1)
        results  = model(frame_rgb)
        print(f'results: {results}')
        detections = results.pandas().xyxy[0].to_dict(orient='records')
        print(f'detections: {detections}')
        if detections:
            for det in detections:
                xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                label = f"{det['name']} {det['confidence']:.2f}"
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv5 Detection', frame)
            cv2.waitKey(1)
    except Exception as e:
        pass

infer("img1.jpg")