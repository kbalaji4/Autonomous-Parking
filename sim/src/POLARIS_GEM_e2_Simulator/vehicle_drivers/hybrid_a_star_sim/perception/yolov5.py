import numpy as np
import cv2
import torch

model_path = 'sim/src/POLARIS_GEM_e2_Simulator/vehicle_drivers/hybrid_a_star_sim/perception/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.conf = 0.7  
    
def infer(img_path):
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (640, 640))
    try:
        cv2.imshow('Original Image', frame)
        cv2.waitKey(1)
        results  = model(frame)
        detections = results.pandas().xyxy[0].to_dict(orient='records')
        print(detections)
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

infer("sim/src/POLARIS_GEM_e2_Simulator/vehicle_drivers/hybrid_a_star_sim/perception/img1.jpg")