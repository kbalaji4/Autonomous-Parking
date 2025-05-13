#!/usr/bin/env python3
import os, sys
import rospy
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_matrix
import message_filters
# import pyzed.sl as sl


scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print("printing paths in main(): ")
print(sys.path)

class YOLOv5Node:
    def __init__(self):
        rospy.init_node('yolov5_detector', anonymous=True)

        # self.zed = sl.Camera()
        # init_params = sl.InitParameters()
        # init_params.camera_resolution = sl.RESOLUTION.HD720
        # init_params.depth_mode        = sl.DEPTH_MODE.ULTRA
        # init_params.coordinate_units  = sl.UNIT.METER
        # if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        #     print("ZED Open Failed"); self.zed.close(); exit()

        # # 2) Turn on ZED’s built-in “world‐frame” SLAM tracker
        # tracking_params = sl.PositionalTrackingParameters()  
        # # (you can set tracking_params.set_as_static=True if ZED is on a fixed tripod)
        # if self.zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
        #     print("ZED enable positional tracking failed"); self.zed.close(); exit()
        
        # Subscribers and Publishers
        self.fl_image_sub = rospy.Subscriber("/camera_fl/arena_camera_node/image_raw", Image, self.fl_image_callback, queue_size=10)
        self.fr_image_sub = rospy.Subscriber("/camera_fr/arena_camera_node/image_raw", Image, self.fr_image_callback, queue_size=10)
        self.rear_image_sub = rospy.Subscriber("/mako_1/mako_1/image_raw", Image, self.rear_image_callback, queue_size=10)
        # self.image_sub = rospy.Subscriber("/oak/right/image_raw",Image, self.image_callback, queue_size=10)
        self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color",Image, self.image_callback, queue_size=10)
        self.depth_sub = rospy.Subscriber("/zed2/zed_node/depth/depth_registered", Image, self.depth_callback, queue_size=10)
        self.camera_info_sub = rospy.Subscriber("/zed2/zed_node/left/camera_info", CameraInfo, self.camera_info_callback, queue_size=10)
        self.camera_pose_sub = rospy.Subscriber("/zed2/zed_node/pose", PoseStamped, self.camera_pose_callback)

        self.annotated_pub = rospy.Publisher("/annotated_image", Image, queue_size=10)
        self.fr_annotated_pub = rospy.Publisher("/fr_annotated_image", Image, queue_size=10)
        self.fl_annotated_pub = rospy.Publisher("/fl_annotated_image", Image, queue_size=10)
        self.rear_annotated_pub = rospy.Publisher("/rear_annotated_image", Image, queue_size=10)
        self.detections_pub = rospy.Publisher("/detections", String, queue_size=1)
        self.fr_detections_pub = rospy.Publisher("/fr_detections", String, queue_size=1)
        self.fl_detections_pub = rospy.Publisher("/fl_detections", String, queue_size=1)
        self.world_coordinates_pub = rospy.Publisher("/detection_world_positions", PoseStamped, queue_size=10)


        self.camera_matrix = None
        self.dist_coeffs = None

        self.camera_position = None
        self.camera_orientation = None

        self.bridge = CvBridge()
        
        # Load model
        self.model_path = '/home/wy16/Desktop/Autonomous-Parking/sim/src/POLARIS_GEM_e2_Simulator/vehicle_drivers/hybrid_a_star_sim/scripts/best.pt'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        # self.model = torch.hub.load("ultralytics/yolov5", "yolov5m") # pretrained coco
        self.model.conf = 0.7

        rospy.loginfo("Model loaded OK")
        
        rospy.loginfo("YOLOv5 Detector Node Initialized.")

    def __del__(self):
        if hasattr(self, 'zed'):
            self.zed.close()

    def camera_pose_callback(self, msg):
        """Store camera position and orientation"""
        self.camera_position = msg.pose.position
        self.camera_orientation = msg.pose.orientation

    def camera_info_callback(self, msg):
        """
        camera_matrix structure:
        [[fx  0  cx]
        [0   fy cy]
        [0   0   1]]
        """
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
        
    # def get_3d_position_zed(self, pixel_x, pixel_y):
    #     """Get 3D position using ZED SDK. so pixel --> camera"""
    #     point_cloud = sl.Mat()
    #     err = self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         return None
            
    #     point3D = sl.float3()
    #     err = point_cloud.get_value(pixel_x, pixel_y, point3D)
    #     if err != sl.ERROR_CODE.SUCCESS:
    #         return None
            
    #     # ZED SDK gives coordinates in camera frame
    #     return [point3D.x, point3D.y, point3D.z]

    # def get_world_transform_zed(self):
    #     """Get current camera pose from ZED SDK. so camera --> world"""
    #     zed_pose = sl.Pose()
    #     if self.zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD) == sl.ERROR_CODE.SUCCESS:
    #         return zed_pose.get_translation(), zed_pose.get_rotation_matrix()
    #     return None

    def get_3d_position(self, pixel_x, pixel_y, depth):
        """
        pixel to camera
        """
        # if self.camera_matrix is None:
        #     return None
            
        # Convert pixel coordinates to 3D coordinates
        cx = self.camera_matrix[0,2]
        cy = self.camera_matrix[1,2]
        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        
        # Calculate X,Y,Z in camera frame
        X = (pixel_x - cx) * depth / fx
        Y = (pixel_y - cy) * depth / fy
        Z = depth
        
        return (X, Y, Z)
    
    def camera_to_world(self, point_camera):
        """Convert point from camera frame to world frame"""
        if self.camera_position is None or self.camera_orientation is None:
            return None
            
        # Extract camera position
        cam_x = self.camera_position.x
        cam_y = self.camera_position.y
        cam_z = self.camera_position.z
        
        # Get camera orientation in euler angles
        quaternion = [
            self.camera_orientation.x,
            self.camera_orientation.y,
            self.camera_orientation.z,
            self.camera_orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quaternion) # or hard coded one?
        
        # Create rotation matrix
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # # Combined rotation matrix
        # R = Rz @ Ry @ Rx

        # could also try this
        R = quaternion_matrix(quaternion)[:3, :3]

        # Camera to world rotation is INVERSE of camera's orientation?
        R_cam_to_world = R.T
        
        # Convert point from camera to world frame
        point_camera_np = np.array(point_camera)
        point_world = R @ point_camera_np + np.array([cam_x, cam_y, cam_z])
        
        return point_world.tolist()

    def publish_world_position(self, world_pos, timestamp):
        """Publish cone position in world coordinates"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = world_pos[0]
        pose_msg.pose.position.y = world_pos[1]
        pose_msg.pose.position.z = world_pos[2]
        pose_msg.pose.orientation.w = 1.0
        self.world_coordinates_pub.publish(pose_msg)

    def depth_callback(self, msg):
        try:
            # Convert depth image to OpenCV format
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")
    
    def fl_image_callback(self, msg):
        """"
        no depth on this one
        """
        try:
            # print(f"fl callback: {msg.header.frame_id}")
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
                # detection_msg.header = msg.header # copy header from image
                self.fl_detections_pub.publish(detection_msg)
                
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
            self.fl_annotated_pub.publish(annotated_img_msg)
        except Exception as e:
            rospy.logerr(f"Error during detection: {e}")
    def fr_image_callback(self, msg):
        """"
        no depth on this one
        """
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
                # detection_msg.header = msg.header # copy header from image
                self.fr_detections_pub.publish(detection_msg)
                
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
            self.fr_annotated_pub.publish(annotated_img_msg)
            
        except Exception as e:
            rospy.logerr(f"Error during detection: {e}")
        
    def rear_image_callback(self, msg):
        """"
        no depth on this one
        """
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
                # detection_msg.header = msg.header # copy header from image
                self.fr_detections_pub.publish(detection_msg)
                
                # Draw bounding boxes on frame
                for det in detections:
                    if det['name'] != 'person':
                        continue
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
            self.rear_annotated_pub.publish(annotated_img_msg)
            
        except Exception as e:
            rospy.logerr(f"Error during detection: {e}")
    
    def image_callback(self, msg):
        try:
            # Convert image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.resize(frame, (640, 640))
            frame = frame[0:641, 100:541] # crop: y same, x cropped by 100
            
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
                for i, det in enumerate(detections):
                    # if det['name'] != 'person':
                    #     continue
                    xmin = int(det['xmin'])
                    ymin = int(det['ymin'])
                    xmax = int(det['xmax'])
                    ymax = int(det['ymax'])

                    # get depth from the bbox
                    depth_roi = self.depth_frame[ymin:ymax, xmin:xmax]
                    if depth_roi.size > 0:
                        # rospy.loginfo(depth_roi)
                        # Calculate the centroid depth or closest depth
                        valid_depths = depth_roi[(depth_roi > 0) & (depth_roi < float('inf'))]  # Exclude invalid depths. inf or nan
                        centroid_depth = float('inf') # by default
                        wx, wy, wz = None, None, None # default
                        position_3d = None
                        if valid_depths.size > 0:
                            centroid_depth = np.median(valid_depths)  # Use median for robustness

                            # x, y is just center of bounding box
                            cone_x = (xmin + xmax) // 2
                            cone_y = (ymin + ymax) // 2
                            rospy.loginfo(f"cone id: {i}, timestamp: {msg.header.stamp}")
                            rospy.loginfo(f"cone detected with depth {centroid_depth:.2f}m")
                            position_3d = self.get_3d_position(cone_x, cone_y, centroid_depth)
                        if position_3d:
                            # """zed2 built in slam"""
                            # # Get world transform
                            # translation, rotation = self.get_world_transform_zed()
                            # if translation and rotation:
                            #     # Transform to world coordinates
                            #     position_3d = np.array(position_3d)
                            #     world_pos = rotation @ position_3d + translation
                                
                            #     # Publish world position
                            #     self.publish_world_position(world_pos.tolist(), msg.header.stamp)
                                
                            #     wx, wy, wz = world_pos
                            #     rospy.loginfo(f"Cone world position: X={wx:.2f}m, Y={wy:.2f}m, Z={wz:.2f}m")
                            
                            """zed2 diy depth"""
                            X, Y, Z = position_3d
                            
                            rospy.loginfo(f"cone camera position: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m")
                            world_pos = self.camera_to_world([X, Y, Z])
                            if world_pos:
                                wx, wy, wz = world_pos
                                # rospy.loginfo(f"person world position: X={wx:.2f}m, Y={wy:.2f}m, Z={wz:.2f}m")
                                self.publish_world_position(world_pos, msg.header.stamp)

                    label = f"{det['name']} {det['confidence']:.2f} {centroid_depth:.2f} {wx:.2f} {wy:.2f} {wz:.2f}"
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