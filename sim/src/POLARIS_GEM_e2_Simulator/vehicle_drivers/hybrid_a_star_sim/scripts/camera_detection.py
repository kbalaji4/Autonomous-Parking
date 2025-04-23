#!/usr/bin/env python3
import os, sys
import rospy
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print("printing paths in main(): ")
print(sys.path)

class YOLOv5Node:
    def __init__(self):
        rospy.init_node('yolov5_detector', anonymous=True)
        
        # Subscribers and Publishers
        self.image_sub = rospy.Subscriber("/oak/left/image_raw",Image, self.image_callback, queue_size=10)
        self.annotated_pub = rospy.Publisher("/annotated_image", Image, queue_size=10)
        self.detections_pub = rospy.Publisher("/detections", String, queue_size=1)
        
        self.bridge = CvBridge()
        
        # Load model
        self.model_path = '/home/wy16/Desktop/Autonomous-Parking/sim/src/POLARIS_GEM_e2_Simulator/vehicle_drivers/hybrid_a_star_sim/scripts/best.pt'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        self.model.conf = 0.7
        
        rospy.loginfo("YOLOv5 Detector Node Initialized.")

    def image_callback(self, msg):
        try:
            # Convert image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.resize(frame, (640, 640))
            
            # Run inference
            results = self.model(frame)
            detections = results.pandas().xyxy[0].to_dict(orient='records')
            
            if detections:
                # rospy.loginfo(f'Detections: {detections}')
                detection_msg = String()
                detection_msg.data = str(detections)
                detection_msg.header = msg.header # copy header from image
                self.detections_pub.publish(detection_msg)
                # self.detections_pub.publish(String(data=str(detections)))
                
                # Draw bounding boxes on frame
                for det in detections:
                    xmin = int(det['xmin'])
                    ymin = int(det['ymin'])
                    xmax = int(det['xmax'])
                    ymax = int(det['ymax'])
                    label = f"{det['name']} {det['confidence']:.2f}"
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert annotated frame back to ROS image and publish
            annotated_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.annotated_pub.publish(annotated_img_msg)
            
        except Exception as e:
            rospy.logerr(f"Error during detection: {e}")

if __name__ == '__main__':
    try:
        node = YOLOv5Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass