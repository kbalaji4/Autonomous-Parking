#!/usr/bin/env python3
import os, sys
import rospy
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
import message_filters

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print("printing paths in main(): ")
print(sys.path)

class YOLOv5Node:
    def __init__(self):
        rospy.init_node('yolov5_detector', anonymous=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers and Publishers
        # self.image_sub = rospy.Subscriber("/zed2/zed_node/left/image_rect_color",Image, self.image_callback, queue_size=10)
        self.image_sub = message_filters.Subscriber("/zed2/zed_node/left/image_rect_color", Image)
        self.pointcloud_sub = message_filters.Subscriber("/zed2/zed_node/point_cloud/cloud_registered", PointCloud2)
        self.camera_info_sub = message_filters.Subscriber("/zed2/zed_node/left/camera_info", CameraInfo)

        self.camera_matrix = None
        self.dist_coeffs = None

        self.ts = message_filters.ApproximateTimeSynchronizer([
            self.image_sub, 
            self.pointcloud_sub, 
            self.camera_info_sub], 
            queue_size=10,
            slop=5.0) # allow some delay
        
        self.tf_timeout = rospy.Duration(0.1)
        self.tf_retries = 3
        
        # Add diagnostic publisher
        self.debug_pub = rospy.Publisher("/yolo_debug", String, queue_size=10)
        
        self.ts.registerCallback(self.synchronized_callback)
        
        self.annotated_pub = rospy.Publisher("/annotated_image", Image, queue_size=10)
        self.detections_pub = rospy.Publisher("/detections", String, queue_size=1)

        self.obstacle_grid_pub = rospy.Publisher("/obstacle_grid", OccupancyGrid, queue_size=1)
        self.obstacle_poses_pub = rospy.Publisher("/obstacle_poses", PoseStamped, queue_size=10)

        self.grid_resolution = 0.1  # meters per cell
        self.grid_width = 200  # cells
        self.grid_height = 200  # cells
        self.obstacle_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)

        self.bridge = CvBridge()
        
        # Load model
        self.model_path = '/home/wy16/Desktop/Autonomous-Parking/sim/src/POLARIS_GEM_e2_Simulator/vehicle_drivers/hybrid_a_star_sim/scripts/best.pt'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        self.model.conf = 0.7
        
        rospy.loginfo("YOLOv5 Detector Node Initialized.")

    # def camera_info_callback(self, msg):
    #     if self.camera_matrix is None:
    #         self.camera_matrix = np.array(msg.K).reshape(3, 3)
    #         self.dist_coeffs = np.array(msg.D)
        
    # def get_3d_position(self, pixel_x, pixel_y, depth):
    #     if self.camera_matrix is None:
    #         return None
            
    #     # Convert pixel coordinates to 3D coordinates
    #     cx = self.camera_matrix[0,2]
    #     cy = self.camera_matrix[1,2]
    #     fx = self.camera_matrix[0,0]
    #     fy = self.camera_matrix[1,1]
        
    #     # Calculate X,Y,Z in camera frame
    #     X = (pixel_x - cx) * depth / fx
    #     Y = (pixel_y - cy) * depth / fy
    #     Z = depth
        
    #     return (X, Y, Z)
    
    def synchronized_callback(self, img_msg, cloud_msg, camera_info_msg):
        try:
            start_time = rospy.Time.now()
            
            # Check if messages are too old
            current_time = rospy.Time.now()
            msg_age = current_time - img_msg.header.stamp
            if msg_age.to_sec() > 1.0:  # If message is older than 1 second
                rospy.logwarn(f"Skipping old messages, age: {msg_age.to_sec():.2f}s")
                return
            
            # Convert image for YOLO detection
            frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            frame = cv2.resize(frame, (640, 640))
            
            # Run YOLO inference
            results = self.model(frame)
            detections = results.pandas().xyxy[0].to_dict(orient='records')
            
            # Process point cloud
            pc_data = pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z"))
            pc_array = np.array(list(pc_data))
            
            detected_obstacles = []
            
            for det in detections:
                xmin, ymin, xmax, ymax = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                
                # Get 3D points within detection bbox
                pc_indices = self.get_points_in_bbox(pc_array, frame, camera_info_msg, xmin, ymin, xmax, ymax)
                if len(pc_indices) > 0:
                    obstacle_points = pc_array[pc_indices]
                    
                    # Calculate centroid of obstacle
                    centroid = np.mean(obstacle_points, axis=0)
                    
                    # Transform from camera frame to map frame
                    obstacle_pose = self.transform_to_map_frame(centroid, img_msg.header.stamp)
                    if obstacle_pose:
                        detected_obstacles.append(obstacle_pose)
                        self.update_obstacle_grid(obstacle_pose)
                        self.obstacle_poses_pub.publish(obstacle_pose)
                    
                    # Draw on image
                    label = f"{det['name']} {det['confidence']:.2f} ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Publish annotated image
            annotated_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.annotated_pub.publish(annotated_img_msg)
            
            # Publish obstacle grid
            self.publish_obstacle_grid()

            processing_time = (rospy.Time.now() - start_time).to_sec()
            self.debug_pub.publish(String(data=f"Processing time: {processing_time:.3f}s"))
            
        except Exception as e:
            rospy.logerr(f"Error in detection pipeline: {e}")

    def get_points_in_bbox(self, pc_array, image, camera_info, xmin, ymin, xmax, ymax):
        """Get indices of 3D points that project into the 2D bounding box"""
        if len(pc_array) == 0:
            return []
        
        # Project 3D points to 2D
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        
        x = pc_array[:, 0]
        y = pc_array[:, 1]
        z = pc_array[:, 2]
        
        # Avoid division by zero
        valid = z > 0.01
        
        u = np.zeros_like(x)
        v = np.zeros_like(y)
        
        try:
            np.divide(x[valid], z[valid], out=u[valid], where=z[valid]!=0)
            np.divide(y[valid], z[valid], out=v[valid], where=z[valid]!=0)
            u[valid] = (fx * u[valid] + cx).astype(int)
            v[valid] = (fy * v[valid] + cy).astype(int)
        except Exception as e:
            rospy.logwarn(f"Error in point projection: {e}")
            return []
        
        # Add bounds checking
        image_height, image_width = image.shape[:2]
        valid &= (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
        
        # Find points within bbox
        in_bbox = (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax) & valid
        
        return np.where(in_bbox)[0]

    def transform_to_map_frame(self, point, stamp):
        """Transform a point from camera frame to map frame with retries"""
        for attempt in range(self.tf_retries):
            try:
                # Wait for transform to become available
                if self.tf_buffer.can_transform("map", "zed2_left_camera_frame", stamp, self.tf_timeout):
                    transform = self.tf_buffer.lookup_transform(
                        "map",
                        "zed2_left_camera_frame",
                        stamp,
                        self.tf_timeout
                    )
                    
                    pose_stamped = PoseStamped()
                    pose_stamped.header.frame_id = "zed2_left_camera_frame"
                    pose_stamped.header.stamp = stamp
                    pose_stamped.pose.position = Point(point[0], point[1], point[2])
                    pose_stamped.pose.orientation.w = 1.0
                    
                    transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
                    return transformed_pose
                else:
                    rospy.logwarn(f"Transform not available on attempt {attempt + 1}/{self.tf_retries}")
                    rospy.sleep(0.05)  # Short delay before retry
                    
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Transform failed on attempt {attempt + 1}/{self.tf_retries}: {e}")
                rospy.sleep(0.05)  # Short delay before retry
                
        rospy.logwarn("Transform failed after all retries")
        return None

    def update_obstacle_grid(self, obstacle_pose):
        """Update occupancy grid with new obstacle position"""
        x = obstacle_pose.pose.position.x
        y = obstacle_pose.pose.position.y
        
        # Convert to grid coordinates
        grid_x = int((x + self.grid_width * self.grid_resolution / 2) / self.grid_resolution)
        grid_y = int((y + self.grid_height * self.grid_resolution / 2) / self.grid_resolution)
        
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            # Update grid with exponential decay
            self.obstacle_grid[grid_y, grid_x] = 100  # Occupied
            self.apply_exponential_decay(grid_x, grid_y)

    def apply_exponential_decay(self, center_x, center_y, decay_factor=0.5):
        """Apply exponential decay around obstacle point"""
        radius = 5  # cells
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x = center_x + dx
                y = center_y + dy
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    distance = np.sqrt(dx**2 + dy**2)
                    value = 100 * np.exp(-decay_factor * distance)
                    self.obstacle_grid[y, x] = max(self.obstacle_grid[y, x], int(value))

    def publish_obstacle_grid(self):
        """Publish occupancy grid"""
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"
        
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height
        grid_msg.info.origin.position.x = -self.grid_width * self.grid_resolution / 2
        grid_msg.info.origin.position.y = -self.grid_height * self.grid_resolution / 2
        
        grid_msg.data = self.obstacle_grid.flatten().tolist()
        self.obstacle_grid_pub.publish(grid_msg)
    
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
                        # rospy.loginfo(depth_roi)
                        # Calculate the centroid depth or closest depth
                        valid_depths = depth_roi[(depth_roi > 0) & (depth_roi < float('inf'))]  # Exclude invalid depths. inf or nan
                        centroid_depth = float('inf') # by default
                        position_3d = None
                        if valid_depths.size > 0:
                            centroid_depth = np.median(valid_depths)  # Use median for robustness
                            cone_x = (xmin + xmax) // 2
                            cone_y = (ymin + ymax) // 2
                            cone_positions.append((cone_x, cone_y, centroid_depth))
                            rospy.loginfo(f"Cone detected with depth {centroid_depth:.2f}m")
                            position_3d = self.get_3d_position(cone_x, cone_y, centroid_depth)
                        if position_3d:
                            X, Y, Z = position_3d
                            rospy.loginfo(f"Cone detected at X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m")
                            
                            # Calculate angle from camera center
                            angle = np.arctan2(X, Z)
                            rospy.loginfo(f"Cone angle from center: {np.degrees(angle):.2f} degrees")

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