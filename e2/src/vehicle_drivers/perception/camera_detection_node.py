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
        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color",Image, self.image_callback, queue_size=10)
        self.annotated_pub = rospy.Publisher("/annotated_image", Image, queue_size=10)
        self.detections_pub = rospy.Publisher("/detections", String, queue_size=1)
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback, queue_size=10)

        self.bridge = CvBridge()
        
        # Load model
        import rospkg
        pkg_path = rospkg.RosPack().get_path('perception')
        self.model_path = os.path.join(pkg_path, 'best.pt')
        rospy.loginfo(f"Loading YOLOv5 weights from {self.model_path}")
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom',
            path=self.model_path,
            force_reload=True
        )
        self.model.conf = 0.7
        rospy.loginfo("Model loaded OK")
        
        rospy.loginfo("YOLOv5 Detector Node Initialized.")

    def depth_callback(self, msg):
        try:
            # Convert depth image to OpenCV format
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
    
    def image_callback(self, msg):
        try:
            # Convert image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.resize(frame, (640, 640))
            
            # Run inference
            results = self.model(frame)
            detections = results.pandas().xyxy[0].to_dict(orient='records')
            
            if detections and hasattr(self, 'depth_frame'):
                # rospy.loginfo(f'Detections: {detections}')
                detection_msg = String()
                detection_msg.data = str(detections)
                # detection_msg.header = msg.header # copy header from image
                self.detections_pub.publish(detection_msg)
                # self.detections_pub.publish(String(data=str(detections)))
                
                # Draw bounding boxes on frame
                for det in detections:
                    cone_positions = []
                    xmin = int(det['xmin'])
                    ymin = int(det['ymin'])
                    xmax = int(det['xmax'])
                    ymax = int(det['ymax'])

                    # get depth from the bbox
                    depth_roi = self.depth_frame[ymin:ymax, xmin:xmax]
                    if depth_roi.size > 0:
                        # Calculate the centroid depth or closest depth
                        valid_depths = depth_roi[depth_roi > 0]  # Exclude invalid depths
                        if valid_depths.size > 0:
                            # centroid_depth = np.median(valid_depths)  # Use median for robustness
                            centroid_depth = valid_depths[0]
                            cone_x = (xmin + xmax) // 2
                            cone_y = (ymin + ymax) // 2
                            cone_positions.append((cone_x, cone_y, centroid_depth))
                            rospy.loginfo(f"Cone detected with depth={centroid_depth:.2f}m")

                    label = f"{det['name']} {det['confidence']:.2f} {centroid_depth:.2f}"
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