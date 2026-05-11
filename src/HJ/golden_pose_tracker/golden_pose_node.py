import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import open3d as o3d
import math
import time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters
import tf2_ros

from scipy.spatial.transform import Rotation as R_scipy

from ultralytics import YOLO

class GoldenPoseTracker(Node):
    def __init__(self):
        super().__init__('golden_pose_tracker')
        self.bridge = CvBridge()
        
        # YOLO Model
        yolo_path = "/home/irol/ros2_ws/src/HJ/yolo/best.pt"
        self.get_logger().info(f"Loading YOLO from {yolo_path}...")
        self.model = YOLO(yolo_path)
        
        # TF Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, '/yolo/golden_boxes', 10)
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            10
        )
        self.K = None
        self.camera_frame = None
        
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_rect_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        
        # Global Memory
        # { box_id: {'pose': [x,y,z, qx,qy,qz,qw], 'ratio': R, 'status': 'visible', 'last_seen': time.time()} }
        self.memory = {}
        self.next_id = 1
        
        # New Detection Lock Feature (Option A)
        self.lock_new_boxes = True       # True: 처음 인식된 상자만 유지, False: 계속 새로운 상자 추가 허용
        self.first_detection_time = None
        self.grace_period = 3.0          # 처음 인식 후 3초 동안만 추가 등록 허용
        
        
        self.get_logger().info("Golden Pose Tracker Initialized. Waiting for camera info and images...")

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.camera_frame = msg.header.frame_id
            self.img_width = msg.width
            self.img_height = msg.height
            self.get_logger().info(f"Camera Info received. Frame: {self.camera_frame}, Resolution: {self.img_width}x{self.img_height}")

    def get_transform(self, target_frame, source_frame):
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return t
        except tf2_ros.TransformException as ex:
            return None

    def transform_points(self, points, t_stamped):
        t = t_stamped.transform.translation
        q = t_stamped.transform.rotation
        
        # Use scipy to convert quaternion to rotation matrix
        r = R_scipy.from_quat([q.x, q.y, q.z, q.w])
        R_mat = r.as_matrix()
        
        trans = np.array([t.x, t.y, t.z])
        points_transformed = (R_mat @ points.T).T + trans
        return points_transformed
        
    def calculate_iou(self, box1, box2):
        # box: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0
            
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def sync_callback(self, color_msg, depth_msg):
        if self.K is None or self.camera_frame is None:
            return
            
        t_base_cam = self.get_transform('base_link', self.camera_frame)
        if t_base_cam is None:
            self.get_logger().warn("Waiting for TF from camera to base_link...")
            return
            
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            # Depth image: 16-bit encoding (mm)
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
            
        results = self.model(cv_color, verbose=False)
        boxes = results[0].boxes
        
        current_frame_boxes = []
        if boxes is not None and len(boxes) > 0:
            boxes_np = boxes.xyxy.cpu().numpy()
            current_frame_boxes = [list(map(int, box[:4])) for box in boxes_np]
            
        # Match detected boxes with memory
        matched_ids = []
        box_to_id = {}
        
        for i, det_box in enumerate(current_frame_boxes):
            best_iou = 0
            best_id = -1
            for mem_id, mem_data in self.memory.items():
                if 'last_bbox' in mem_data and mem_id not in matched_ids:
                    iou = self.calculate_iou(det_box, mem_data['last_bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_id = mem_id
            
            if best_iou > 0.3:
                box_to_id[i] = best_id
                matched_ids.append(best_id)
            else:
                # --- 새로운 박스가 발견되었을 때 ---
                if self.first_detection_time is None:
                    self.first_detection_time = time.time() # 최초 발견 시간 기록
                    self.get_logger().info(f"First box detected! Grace period started ({self.grace_period}s).")

                # 기능이 켜져 있고, 유예 시간(3초)이 지났다면 등록 거부
                if self.lock_new_boxes and (time.time() - self.first_detection_time > self.grace_period):
                    box_to_id[i] = None
                else:
                    box_to_id[i] = self.next_id
                    matched_ids.append(self.next_id)
                    self.next_id += 1
                
        # Update memory status for all
        for mem_id in self.memory.keys():
            if mem_id not in matched_ids:
                self.memory[mem_id]['status'] = 'memorized'
                
        # Process detections
        for i, det_box in enumerate(current_frame_boxes):
            box_id = box_to_id[i]
            x1, y1, x2, y2 = det_box
            
            # Draw bounding boxes for visualization
            if box_id is None:
                cv2.rectangle(cv_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(cv_color, "Ignored", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue
            else:
                cv2.rectangle(cv_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_color, f"ID: {box_id}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            w = x2 - x1
            h = y2 - y1
            
            if w == 0 or h == 0:
                continue
                
            # Check if touching boundary (margin 5 pixels)
            margin = 5
            touching_left = x1 <= margin
            touching_right = x2 >= self.img_width - margin
            touching_top = y1 <= margin
            touching_bottom = y2 >= self.img_height - margin
            is_clipped = touching_left or touching_right or touching_top or touching_bottom
            
            # --- 1. Initial Registration (Not clipped, good view) ---
            if not is_clipped:
                # Safe ROI: 10% shrink
                shrink_x = int(w * 0.1)
                shrink_y = int(h * 0.1)
                safe_x1 = x1 + shrink_x
                safe_y1 = y1 + shrink_y
                safe_x2 = x2 - shrink_x
                safe_y2 = y2 - shrink_y
                
                # Extract depth for ROI
                roi_depth = cv_depth[safe_y1:safe_y2, safe_x1:safe_x2]
                valid_mask = (roi_depth > 0) & (roi_depth < 3000) # Valid depth < 3m
                
                if np.sum(valid_mask) > 100: # Need enough points for RANSAC
                    # Create U, V meshgrid for the ROI
                    u, v = np.meshgrid(np.arange(safe_x1, safe_x2), np.arange(safe_y1, safe_y2))
                    u_valid = u[valid_mask]
                    v_valid = v[valid_mask]
                    z_valid = roi_depth[valid_mask] / 1000.0 # Convert to meters
                    
                    # 2D to 3D projection in camera frame
                    fx, fy = self.K[0,0], self.K[1,1]
                    cx, cy = self.K[0,2], self.K[1,2]
                    
                    x_cam = (u_valid - cx) * z_valid / fx
                    y_cam = (v_valid - cy) * z_valid / fy
                    
                    points_cam = np.vstack((x_cam, y_cam, z_valid)).T
                    
                    # Transform to base_link
                    points_base = self.transform_points(points_cam, t_base_cam)
                    
                    # RANSAC Plane Fitting
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_base)
                    pcd.estimate_normals()
                    
                    # distance_threshold: 1cm
                    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                             ransac_n=3,
                                                             num_iterations=1000)
                    [a, b, c, d] = plane_model
                    
                    inlier_cloud = pcd.select_by_index(inliers)
                    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
                    
                    # Normal vector (Ensure it points away from base_link / outwards)
                    # Generally base_link is at the bottom, so normal points to camera
                    cam_pos = np.array([t_base_cam.transform.translation.x, 
                                        t_base_cam.transform.translation.y, 
                                        t_base_cam.transform.translation.z])
                    normal = np.array([a, b, c])
                    if np.dot(normal, cam_pos - centroid) < 0:
                        normal = -normal
                        
                    # Calculate orientation from normal
                    # Assume Z-axis is normal, Y-axis is world Z projected onto plane
                    z_axis = normal / np.linalg.norm(normal)
                    up_vector = np.array([0, 0, 1])
                    if abs(np.dot(up_vector, z_axis)) > 0.99:
                        up_vector = np.array([0, 1, 0])
                    x_axis = np.cross(up_vector, z_axis)
                    x_axis /= np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    
                    R_mat = np.column_stack([x_axis, y_axis, z_axis])
                    r_scipy = R_scipy.from_matrix(R_mat)
                    qx, qy, qz, qw = r_scipy.as_quat()
                    
                    # Save Golden Pose
                    golden_ratio = float(w) / float(h)
                    
                    self.memory[box_id] = {
                        'pose': [centroid[0], centroid[1], centroid[2], qx, qy, qz, qw],
                        'ratio': golden_ratio,
                        'status': 'visible',
                        'last_seen': time.time(),
                        'last_bbox': det_box,
                        'width_px': w,
                        'height_px': h
                    }
                    
            # --- 2. Continuous Reconstruction (Clipped, use Ratio) ---
            elif is_clipped and box_id in self.memory:
                mem_data = self.memory[box_id]
                R = mem_data['ratio']
                
                # Estimate true center in 2D
                true_w, true_h = w, h
                cx_2d = (x1 + x2) / 2.0
                cy_2d = (y1 + y2) / 2.0
                
                # If touching left/right, width is likely cut
                if touching_left or touching_right:
                    true_w = h * R
                    if touching_left:
                        cx_2d = x2 - (true_w / 2.0)
                    else:
                        cx_2d = x1 + (true_w / 2.0)
                        
                # If touching top/bottom, height is likely cut
                if touching_top or touching_bottom:
                    true_h = w / R
                    if touching_top:
                        cy_2d = y2 - (true_h / 2.0)
                    else:
                        cy_2d = y1 + (true_h / 2.0)
                
                # Sample depth at current visible center
                vc_x = int((x1 + x2) / 2)
                vc_y = int((y1 + y2) / 2)
                # Sample a small patch to get average depth
                patch_depth = cv_depth[max(0, vc_y-5):min(self.img_height, vc_y+5), 
                                       max(0, vc_x-5):min(self.img_width, vc_x+5)]
                valid_patch = patch_depth[patch_depth > 0]
                
                if len(valid_patch) > 0:
                    avg_z = np.mean(valid_patch) / 1000.0
                    
                    # Project True 2D center to 3D Camera Frame
                    fx, fy = self.K[0,0], self.K[1,1]
                    cam_cx, cam_cy = self.K[0,2], self.K[1,2]
                    
                    true_cam_x = (cx_2d - cam_cx) * avg_z / fx
                    true_cam_y = (cy_2d - cam_cy) * avg_z / fy
                    
                    pts_cam = np.array([[true_cam_x, true_cam_y, avg_z]])
                    pts_base = self.transform_points(pts_cam, t_base_cam)[0]
                    
                    # Update Pose only for position (keep locked orientation)
                    old_pose = mem_data['pose']
                    new_pose = [pts_base[0], pts_base[1], pts_base[2], old_pose[3], old_pose[4], old_pose[5], old_pose[6]]
                    
                    self.memory[box_id]['pose'] = new_pose
                    self.memory[box_id]['status'] = 'visible'
                    self.memory[box_id]['last_seen'] = time.time()
                    self.memory[box_id]['last_bbox'] = det_box
            
            else:
                # Clipped but no memory exists (can't reconstruct without Golden Ratio)
                pass
                
        self.publish_tfs_and_markers(color_msg.header.stamp)
        
        # Display real-time view
        cv2.imshow("YOLO Real-time View", cv_color)
        cv2.waitKey(1)

    def publish_tfs_and_markers(self, stamp):
        marker_array = MarkerArray()
        
        # Delete old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        for mem_id, data in self.memory.items():
            pose = data['pose']
            status = data['status']
            
            # TF Broadcast
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = 'base_link'
            t.child_frame_id = f'box_{mem_id}'
            
            t.transform.translation.x = float(pose[0])
            t.transform.translation.y = float(pose[1])
            t.transform.translation.z = float(pose[2])
            t.transform.rotation.x = float(pose[3])
            t.transform.rotation.y = float(pose[4])
            t.transform.rotation.z = float(pose[5])
            t.transform.rotation.w = float(pose[6])
            
            self.tf_broadcaster.sendTransform(t)
            
            # Marker Broadcast
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = stamp
            marker.ns = 'golden_pose'
            marker.id = mem_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(pose[0])
            marker.pose.position.y = float(pose[1])
            marker.pose.position.z = float(pose[2])
            marker.pose.orientation.x = float(pose[3])
            marker.pose.orientation.y = float(pose[4])
            marker.pose.orientation.z = float(pose[5])
            marker.pose.orientation.w = float(pose[6])
            
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Green if visible, Yellow if memorized
            if status == 'visible':
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.8
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.5
                
            # Lifetime = 0 means forever, but we can manage them through tracking
            marker.lifetime = rclpy.duration.Duration().to_msg() 
            
            marker_array.markers.append(marker)
            
        if len(marker_array.markers) > 1:
            self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = GoldenPoseTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
