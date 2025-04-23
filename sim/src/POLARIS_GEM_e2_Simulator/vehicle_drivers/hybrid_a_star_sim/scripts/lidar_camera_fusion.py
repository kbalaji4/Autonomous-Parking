#!/usr/bin/env python3
import rospy
import numpy as np
import message_filters
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import String
from geometry_msgs.msg import Point, PointStamped
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import json
import cv2
import os
import sys

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

print("printing paths in main(): ")
print(sys.path)

class LidarCameraFusion:
    def __init__(self):
        rospy.init_node('lidar_camera_fusion')
        
        # Transform buffer for coordinate conversions
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Camera calibration info
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = None
        self.image_height = None
        
        # Publishers
        self.marker_pub = rospy.Publisher('/fusion/obstacles', MarkerArray, queue_size=1)
        self.visual_pub = rospy.Publisher('/fusion/visual', Image, queue_size=1)

        # handle transforms for fusion
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Synchronize LiDAR and camera detections
        self.bridge = CvBridge()
        lidar_sub = message_filters.Subscriber('/ouster/points', PointCloud2)
        detect_sub = message_filters.Subscriber('/detections', String)
        camera_sub = message_filters.Subscriber('/oak/left/image_raw', Image)
        camera_info_sub = rospy.Subscriber('/oak/left/camera_info', CameraInfo, self.camera_info_callback)
        
        # Synchronize messages with a time tolerance of 0.1 seconds
        sync = message_filters.ApproximateTimeSynchronizer(
            [lidar_sub, detect_sub, camera_sub], 10, 0.5)
        sync.registerCallback(self.fusion_callback)



    def camera_info_callback(self, msg):
        """ 
        This gets our calibration info, nice
        """
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.image_width = msg.width
            self.image_height = msg.height

    def project_point_to_image(self, point_3d):
        """
        project 3D point to image plane
        """
        if self.camera_matrix is None:
            return None
            
        point_2d = cv2.projectPoints(
            np.array([point_3d]),
            np.zeros(3),  # assumes camera is identity frame, it may not be
            np.zeros(3),
            self.camera_matrix,
            self.dist_coeffs
        )[0].reshape(-1)
        
        return point_2d

    def points_in_bbox(self, points, bbox, margin=0.5):
        """Check if points fall within 2D bounding box with margin"""
        xmin, ymin, xmax, ymax = bbox
        points_2d = []
        
        for point in points:
            point_2d = self.project_point_to_image(point)
            if point_2d is None:
                continue
                
            x, y = point_2d
            if (xmin - margin <= x <= xmax + margin and 
                ymin - margin <= y <= ymax + margin):
                points_2d.append((point, (x, y)))
                
        return points_2d
    
    def transform_points_to_camera_frame(self, points, timestamp):
        """Transform points from os_sensor to oak_left_camera_optical_frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'oak_left_camera_optical_frame',  # target frame
                'os_sensor',                      # source frame
                timestamp,
                rospy.Duration(0.1)
            )
            
            # Convert points to camera frame
            transformed_points = []
            for point in points:
                p = PointStamped() # from geometry_msgs.msg
                p.header.frame_id = 'os_sensor'
                p.header.stamp = timestamp
                p.point.x = point[0]
                p.point.y = point[1]
                p.point.z = point[2]
                
                transformed = tf2_geometry_msgs.do_transform_point(p, transform)
                transformed_points.append([
                    transformed.point.x,
                    transformed.point.y,
                    transformed.point.z
                ])
            
            return np.array(transformed_points)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f'Transform failed: {e}')
            return None

    def filter_points_in_view(self, points_camera_frame, max_distance=20.0):
        """Filter points that are in front of camera and within max distance"""
        # Points must be in camera frame
        mask = (points_camera_frame[:, 2] > 0) & \
               (points_camera_frame[:, 2] < max_distance)
        return points_camera_frame[mask]

    def get_points_in_bbox3d(self, points_camera_frame, bbox2d, margin=0.5):
        """Get points within 3D projection of 2D bounding box"""
        xmin, ymin, xmax, ymax = bbox2d
        
        # Project 3D points to image plane
        points_2d = []
        valid_points = []
        
        for i, point in enumerate(points_camera_frame):
            if point[2] <= 0:  # Behind camera
                continue
                
            # Project point
            x = point[0] / point[2] * self.camera_matrix[0, 0] + self.camera_matrix[0, 2]
            y = point[1] / point[2] * self.camera_matrix[1, 1] + self.camera_matrix[1, 2]
            
            # Check if point projects into bbox
            if (xmin - margin <= x <= xmax + margin and 
                ymin - margin <= y <= ymax + margin):
                points_2d.append((int(x), int(y)))
                valid_points.append(point)
                
        return np.array(valid_points), points_2d
    
    def fusion_callback(self, lidar_msg, detections_msg, image_msg):
        try:
            rospy.loginfo("Received synchronized messages")
            # Convert camera image
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # Convert pointcloud to numpy array
            points = []
            for p in pc2.read_points(lidar_msg, skip_nans=True):
                points.append([p[0], p[1], p[2]])
            points = np.array(points)
            
            # Transform points to camera frame
            points_camera = self.transform_points_to_camera_frame(points, lidar_msg.header.stamp)
            if points_camera is None:
                return
                
            # Filter points in view
            points_in_view = self.filter_points_in_view(points_camera)

            # Parse detections
            detections = eval(detections_msg.data) # string to dict
            print(f"Detections in fusion: {detections}")

            # Process each detection
            marker_array = MarkerArray()
            for i, det in enumerate(detections):
                # Get bounding box
                rospy.loginfo(f"Processing detection {i}: bbox={bbox}")
                bbox = [det['xmin'], det['ymin'], det['xmax'], det['ymax']]
                rospy.loginfo(f"Found {len(bbox_points)} points in bbox {i}")
                # Find points within bbox
                bbox_points, points_2d = self.get_points_in_bbox3d(points_in_view, bbox)
                # bbox_points = self.points_in_bbox(points, bbox)
                if not bbox_points or len(bbox_points) < 3:
                    # doesn't exist OR too few points
                    continue
                print(f'Points in bbox: {bbox_points}')

                # we want to get the closest point to the camera, we'll see if it works?
                distances = np.linalg.norm(bbox_points, axis=1)
                closest_idx = np.argmin(distances)
                centroid = bbox_points[closest_idx]

                # # Calculate centroid of points
                # lidar_points = np.array([p[0] for p in bbox_points])
                # centroid = np.mean(lidar_points, axis=0)
                
                # Create marker
                marker = Marker()
                marker.header = lidar_msg.header
                marker.ns = "fusion"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = centroid[0]
                marker.pose.position.y = centroid[1]
                marker.pose.position.z = centroid[2]
                marker.scale.x = marker.scale.y = marker.scale.z = 0.5
                marker.color.r = 1.0
                marker.color.a = 0.8
                marker_array.markers.append(marker)
                
                # draw the bounding box lidar points on the image
                for p2d in points_2d:
                    cv2.circle(image, p2d, 2, (0, 255, 0), -1)
                # for _, point_2d in bbox_points:
                #     x, y = map(int, point_2d)
                #     cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                
                # Draw centroid
                centroid_2d = (
                    int(centroid[0] / centroid[2] * self.camera_matrix[0, 0] + self.camera_matrix[0, 2]),
                    int(centroid[1] / centroid[2] * self.camera_matrix[1, 1] + self.camera_matrix[1, 2])
                )
                cv2.circle(image, centroid_2d, 5, (0, 0, 255), -1)
                cv2.putText(image, f"d={centroid[2]:.1f}m", 
                          (centroid_2d[0], centroid_2d[1]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # centroid_2d = self.project_point_to_image(centroid)
                # if centroid_2d is not None:
                #     x, y = map(int, centroid_2d)
                #     cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                #     cv2.putText(image, f"d={centroid[2]:.1f}m", (x, y-10),
                #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Publish results
            self.marker_pub.publish(marker_array)
            self.visual_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Fusion error: {str(e)}")

if __name__ == '__main__':
    try:
        node = LidarCameraFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass