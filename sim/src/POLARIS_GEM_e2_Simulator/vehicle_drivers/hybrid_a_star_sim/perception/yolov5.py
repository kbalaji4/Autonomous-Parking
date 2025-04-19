import numpy as np
from cv_bridge import CvBridge
import cv2
import torch

def __init__(self):
    super().__init__('YOLOv5')
    self.model_path = 'best.pt' 
    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
    self.model.conf = 0.7  
    
def infer(self,img_path):
    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (640, 640))
    try:
        results = self.model(frame)
        detections = results.pandas().xyxy[0].to_dict(orient='records')
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