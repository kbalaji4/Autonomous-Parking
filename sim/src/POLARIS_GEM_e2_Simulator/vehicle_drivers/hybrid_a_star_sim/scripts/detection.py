import numpy as np
from cv_bridge import CvBridge
import cv2
import torch

from std_msgs.msg import String
from sensor_msgs.msg import Image
from detection_msgs.msg import BoundingBox, BoundingBoxes

class YOLOv5(Node):
    def __init__(self):
        super().__init__('YOLOv5')
        
        self.image_subscriber = self.create_subscription(Image, "camera", self.image_callback, 10)
        self.detection_publisher = self.create_publisher(BoundingBoxes, "detections", 10)
        self.image_publisher = self.create_publisher(Image, "annotated_image", 10)
        self.model_path = 'best.pt' 
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        self.model.conf = 0.7  
        self.bridge = CvBridge()
        
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.resize(frame, (640, 640))
            results = self.model(frame)
            detections = results.pandas().xyxy[0].to_dict(orient='records')
            if detections:
                self.get_logger().info(f'Detections: {detections}')
                self.detection_publisher.publish(String(data=str(detections)))
                bounding_boxes_msg = BoundingBoxes()
                for det in detections:
                    bbox = BoundingBox()
                    bbox.xmin = det['xmin']
                    bbox.ymin = det['ymin']
                    bbox.xmax = det['xmax']
                    bbox.ymax = det['ymax']
                    bbox.probability = det['confidence']
                    bbox.class_id = int(det['class'])
                    bbox.class_name = det['name']
                    bounding_boxes_msg.bounding_boxes.append(bbox)
                    
                    xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    label = f"{det['name']} {det['confidence']:.2f}"
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
                #cv2.imshow('YOLOv5 Detection', frame)
                #cv2.waitKey(1)
                
                self.detection_publisher.publish(bounding_boxes_msg)
                annotated_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.image_publisher.publish(annotated_image_msg)

        except Exception as e:
            self.get_logger().error(f"Error during detection: {e}")

def main(args=None):
    rclpy.init(args=args)
    yolov5_node = YOLOv5()
    rclpy.spin(yolov5_node)
    yolov5_node.destroy_node()
    rclpy.shutdown()
    
    
