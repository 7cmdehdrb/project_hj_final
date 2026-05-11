import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf2_ros
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
import os
from cv_bridge import CvBridge
import message_filters

import cv2
import numpy as np
import math

from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.utils.plotting import Annotator, colors

class SAM3Node(Node):
    def __init__(self):
        super().__init__('sam3_node')
        self.bridge = CvBridge()
        
        # State variables for saving TFs
        self.latest_box_tfs = []
        self.latest_rotated_tfs = []
        self.latest_landmark_tfs = []
        
        # Initialize SAM3 predictors
        sam3_model_path = "/home/irol/ros2_ws/src/HJ/2. SAM3/sam3.pt"
        self.get_logger().info(f"Loading SAM3 model from {sam3_model_path} ...")
        overrides = dict(conf=0.50, task="segment", mode="predict", model=sam3_model_path, verbose=False)
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.predictor2 = SAM3SemanticPredictor(overrides=overrides)
        self.predictor2.setup_model()
        
        # RViz Maker Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/sam3/box_markers', 10)
        
        # TF Broadcaster & Listener
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Load and publish landmarks
        self.load_and_publish_landmarks()
        
        # Configure subscriptions with message_filters
        self.image_sub = message_filters.Subscriber(self, Image, '/virtual_camera/image')
        self.xyz_sub = message_filters.Subscriber(self, Image, '/virtual_camera/xyz_map')
        
        # ApproximateTimeSynchronizer allows small delay between topics
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.xyz_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        
        self.get_logger().info("SAM3 Node Initialized. Waiting for synchronized images...")

    def load_and_publish_landmarks(self):
        filename = '/home/irol/ros2_ws/src/HJ/saved_landmarks.txt'
        if not os.path.exists(filename):
            self.get_logger().warn(f"Landmark file not found: {filename}")
            return
            
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            self.get_logger().error(f"Failed to read landmarks: {e}")
            return
            
        static_transforms = []
        now = self.get_clock().now().to_msg()
        
        for line in lines:
            line = line.strip()
            if not line.startswith("ID:"):
                continue
            
            parts = [p.strip() for p in line.split('|')]
            data = {}
            for part in parts:
                k_v = part.split(':', 1)
                if len(k_v) >= 2:
                    data[k_v[0].strip()] = k_v[1].strip()
                    
            if all(k in data for k in ['ID', 'X', 'Y', 'Z', 'oX', 'oY', 'oZ', 'oW']):
                t = TransformStamped()
                t.header.stamp = now
                t.header.frame_id = data.get('Frame', 'map')
                t.child_frame_id = f"landmark_{data['ID']}"
                
                t.transform.translation.x = float(data['X'])
                t.transform.translation.y = float(data['Y'])
                t.transform.translation.z = float(data['Z'])
                
                t.transform.rotation.x = float(data['oX'])
                t.transform.rotation.y = float(data['oY'])
                t.transform.rotation.z = float(data['oZ'])
                t.transform.rotation.w = float(data['oW'])
                
                static_transforms.append(t)
                
        if static_transforms:
            self.static_tf_broadcaster.sendTransform(static_transforms)
            self.latest_landmark_tfs = static_transforms
            self.get_logger().info(f"Published {len(static_transforms)} landmarks as static TFs.")

    def rotation_matrix_to_quaternion(self, R):
        """3x3 회전 행렬을 쿼터니언 (x, y, z, w)으로 변환"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return x, y, z, w

    def sync_callback(self, img_msg, xyz_msg):
        # Convert ROS2 messages to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            xyz_map = self.bridge.imgmsg_to_cv2(xyz_msg, desired_encoding='32FC3') # (H, W, 3) float32
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
            
        try:
            t_map_cam = self.tf_buffer.lookup_transform(
                'map',
                'virtual_camera_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            cam_pos = np.array([
                t_map_cam.transform.translation.x,
                t_map_cam.transform.translation.y,
                t_map_cam.transform.translation.z
            ])
        except Exception as e:
            # self.get_logger().warn(f"TF lookup failed: {e}")
            cam_pos = np.array([-0.5, 0.0, 0.35]) # Fallback static position if TF not available
            
        src_shape = cv_image.shape[:2]

        # Extract features from the current image
        self.predictor.set_image(cv_image)
        
        # Perform inference using text prompt "box"
        masks, boxes = self.predictor2.inference_features(self.predictor.features, src_shape=src_shape, text=["front face of the box"])
        
        if masks is not None:
            masks_np = masks.cpu().numpy()
            boxes_np = boxes.cpu().numpy()
            
            self.latest_box_tfs = []
            self.latest_rotated_tfs = []
            
            marker_array = MarkerArray()
            # Delete all previous markers to prevent ghost boxes
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            
            # Print bounding box & Map Pose
            for i, box in enumerate(boxes_np):
                # 2D Bounding Box 좌표 추출 (원본)
                orig_x1, orig_y1, orig_x2, orig_y2 = map(int, box[:4])
                
                # 경계선 노이즈(값이 튀는 현상) 방지를 위해 Bounding Box를 안쪽으로 축소 (가로/세로 15% 마진)
                margin_ratio = 0.2
                box_w = orig_x2 - orig_x1
                box_h = orig_y2 - orig_y1
                
                x1 = int(orig_x1 + box_w * margin_ratio)
                y1 = int(orig_y1 + box_h * margin_ratio)
                x2 = int(orig_x2 - box_w * margin_ratio)
                y2 = int(orig_y2 - box_h * margin_ratio)
                
                # 축소된 박스가 너무 작아지는 예외적인 경우엔 원본 좌표로 복구
                if x2 <= x1 or y2 <= y1:
                    x1, y1, x2, y2 = orig_x1, orig_y1, orig_x2, orig_y2

                # 이미지 밖으로 벗어나지 않도록 범위 클리핑
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(xyz_map.shape[1], x2), min(xyz_map.shape[0], y2)
                
                # Bounding box (축소된) 영역의 3D 포인트들 추출
                points_3d = xyz_map[y1:y2, x1:x2].reshape(-1, 3)
                
                # Filter out NaNs (where depth/xyz was invalid)
                valid_points = points_3d[~np.isnan(points_3d).any(axis=1)]
                
                if len(valid_points) > 3:
                    # --- 1. RANSAC을 이용한 평면 피팅 및 노이즈 제거 (Numpy 기반) ---
                    num_points = len(valid_points)
                    best_inliers = []
                    
                    if num_points > 10:
                        max_iterations = 500
                        distance_threshold = 0.01 # 2cm (노이즈 허용 범위)
                        
                        for _ in range(max_iterations):
                            # 무작위 3점 샘플링
                            sample_idx = np.random.choice(num_points, 3, replace=False)
                            p1, p2, p3 = valid_points[sample_idx]
                            
                            # 평면 법선 계산
                            v1 = p2 - p1
                            v2 = p3 - p1
                            normal = np.cross(v1, v2)
                            norm_len = np.linalg.norm(normal)
                            if norm_len < 1e-6:
                                continue
                            normal = normal / norm_len
                            
                            # 평면 방정식 (ax + by + cz + d = 0)
                            d = -np.dot(normal, p1)
                            
                            # 모든 점들과 평면 사이의 거리 계산
                            distances = np.abs(np.dot(valid_points, normal) + d)
                            inliers = np.where(distances < distance_threshold)[0]
                            
                            # 가장 inlier가 많은 평면 선택
                            if len(inliers) > len(best_inliers):
                                best_inliers = inliers
                                
                    # 노이즈(Outlier)를 제거한 Inlier 점들만 필터링
                    if len(best_inliers) > 10:
                        valid_points = valid_points[best_inliers]

                    # --- 2. Inlier 점들만을 사용해 최종 중심과 방향(PCA) 계산 ---
                    # Calculate centroid (mean position)
                    centroid = np.mean(valid_points, axis=0)
                    
                    # PCA (주성분 분석)를 통한 orientation 계산
                    centered = valid_points - centroid
                    cov = np.cov(centered.T) # 3x3 공분산 행렬
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    
                    # 고유값을 내림차순으로 정렬
                    sort_idx = np.argsort(eigenvalues)[::-1]
                    eigenvectors = eigenvectors[:, sort_idx]
                    
                    # 1. Surface Normal (가장 분산이 작은 3번째 축을 상자의 정면(Z축)으로 사용)
                    uz = eigenvectors[:, 2].copy()
                    
                    # 법선 벡터가 항상 카메라 쪽을 바라보도록 부호 일치 방지
                    if np.dot(uz, cam_pos - centroid) < 0:
                        uz = -uz

                    # 2. 가로/세로 길이에 따라 v0, v1이 바뀌는 문제 해결 (수직/수평 판별)
                    v0 = eigenvectors[:, 0].copy()
                    v1 = eigenvectors[:, 1].copy()
                    
                    # 두 벡터 중 수직 성분(지면 기준 Z값)이 더 큰 것을 상자의 위쪽(Y축 또는 Up)으로 설정
                    if abs(v0[2]) > abs(v1[2]):
                        uy = v0
                    else:
                        uy = v1
                        
                    # 위쪽(+Z)을 향하도록 부호 일치
                    if uy[2] < 0:
                        uy = -uy
                        
                    # 3. 우수 좌표계를 따르게끔 외적(Cross product)으로 나머지 X축 생성
                    ux = np.cross(uy, uz)
                    ux_norm = np.linalg.norm(ux)
                    if ux_norm > 1e-6:
                        ux = ux / ux_norm
                    else:
                        # 혹시 모를 예외에 대한 안전 장치
                        ux = np.array([1.0, 0.0, 0.0])
                    
                    # 일관된(축 및 부호가 고정된) 회전 행렬 완성
                    R = np.column_stack([ux, uy, uz])
                        
                    # 회전 쿼터니언 변환
                    qx, qy, qz, qw = self.rotation_matrix_to_quaternion(R)
                    
                    # Euler 변환 (Degrees)
                    t0 = +2.0 * (qw * qx + qy * qz)
                    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
                    roll_deg = math.degrees(math.atan2(t0, t1))
                    
                    t2 = +2.0 * (qw * qy - qz * qx)
                    t2 = +1.0 if t2 > +1.0 else t2
                    t2 = -1.0 if t2 < -1.0 else t2
                    pitch_deg = math.degrees(math.asin(t2))
                    
                    t3 = +2.0 * (qw * qz + qx * qy)
                    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
                    yaw_deg = math.degrees(math.atan2(t3, t4))

                    self.get_logger().info(f"Detected Box {i+1}:")
                    # self.get_logger().info(f"  - 2D BBox: {np.round(box[:4], 1)}")
                    self.get_logger().info(f"  - 3D Map Pose [X, Y, Z]: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
                    self.get_logger().info(f"  - 3D Euler [Roll, Pitch, Yaw] (deg): [{roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}]")
                    
                    # Create RViz Marker
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = img_msg.header.stamp
                    marker.ns = "sam3_boxes"
                    marker.id = i
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    marker.pose.position.x = float(centroid[0])
                    marker.pose.position.y = float(centroid[1])
                    marker.pose.position.z = float(centroid[2])
                    
                    marker.pose.orientation.x = qx
                    marker.pose.orientation.y = qy
                    marker.pose.orientation.z = qz
                    marker.pose.orientation.w = qw
                    
                    # Box size visualization (can be arbitrary, or use bounding box 3d dimensions)
                    ranges = np.ptp(valid_points, axis=0) if len(valid_points) > 1 else [0.2, 0.2, 0.2]
                    marker.scale.x = max(0.1, float(ranges[0]))
                    marker.scale.y = max(0.1, float(ranges[1]))
                    marker.scale.z = max(0.1, float(ranges[2]))
                    
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.5 # Semi-transparent
                    
                    marker.lifetime.sec = 1 # disappear after 1 second if not updated
                    
                    marker_array.markers.append(marker)
                    
                    # --- TF Broadcast 추가 ---
                    t = TransformStamped()
                    t.header.stamp = img_msg.header.stamp
                    t.header.frame_id = "map"
                    # 식별하기 쉽게 Box 인덱스를 TF 이름으로 사용
                    t.child_frame_id = f"sam3_box_{i+1}"
                    
                    t.transform.translation.x = float(centroid[0])
                    t.transform.translation.y = float(centroid[1])
                    t.transform.translation.z = float(centroid[2])
                    
                    # 주성분 분석(PCA)으로 구한 orientation 적용
                    t.transform.rotation.x = qx
                    t.transform.rotation.y = qy
                    t.transform.rotation.z = qz
                    t.transform.rotation.w = qw
                    self.tf_broadcaster.sendTransform(t)
                    
                    # --- 회전된 TF Broadcast 추가 (로컬 Z축 -90도 -> 로컬 Y축 -90도 회전) ---
                    R_z_m90 = np.array([
                        [ 0.0,  1.0, 0.0],
                        [-1.0,  0.0, 0.0],
                        [ 0.0,  0.0, 1.0]
                    ])
                    R_y_m90 = np.array([
                        [ 0.0,  0.0, -1.0],
                        [ 0.0,  1.0,  0.0],
                        [ 1.0,  0.0,  0.0]
                    ])
                    # 연속 회전 적용 (Map 기준 Absolute Rotation)
                    R_global_rot = R @ R_z_m90 @ R_y_m90
                    qx_rot, qy_rot, qz_rot, qw_rot = self.rotation_matrix_to_quaternion(R_global_rot)
                    
                    t_rot = TransformStamped()
                    t_rot.header.stamp = img_msg.header.stamp
                    # Map을 부모로 설정하여 절대 위치(Position)까지 포함하도록 변경
                    t_rot.header.frame_id = "map"
                    t_rot.child_frame_id = f"sam3_box_{i+1}_rotated"
                    
                    t_rot.transform.translation.x = float(centroid[0])
                    t_rot.transform.translation.y = float(centroid[1])
                    t_rot.transform.translation.z = float(centroid[2])
                    
                    t_rot.transform.rotation.x = qx_rot
                    t_rot.transform.rotation.y = qy_rot
                    t_rot.transform.rotation.z = qz_rot
                    t_rot.transform.rotation.w = qw_rot
                    
                    self.tf_broadcaster.sendTransform(t_rot)
                    
                    self.latest_box_tfs.append(t)
                    self.latest_rotated_tfs.append(t_rot)
                    
                else:
                    self.get_logger().warn(f"Detected Box {i+1}: No valid 3D points found inside the mask.")
            
            if len(marker_array.markers) > 1: # only DELETEALL is not enough
                self.marker_pub.publish(marker_array)
            
            # Visualize results
            annotator = Annotator(cv_image, pil=False)
            annotator.masks(masks_np, [colors(x, True) for x in range(len(masks_np))])
            result_img = annotator.result()
            
            # cv2.imshow("SAM3 Bounding Box & Masks", result_img)
            # cv2.waitKey(1)
            
        else:
            self.get_logger().info("No 'box' detected.")
            # cv2.imshow("SAM3 Bounding Box & Masks", cv_image)
            # cv2.waitKey(1)

    def save_poses_to_file(self):
        filename = '/home/irol/ros2_ws/src/HJ/sam3_final_poses.txt'
        
        def quat_to_euler(q):
            t0 = +2.0 * (q.w * q.x + q.y * q.z)
            t1 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            roll = math.degrees(math.atan2(t0, t1))
            t2 = +2.0 * (q.w * q.y - q.z * q.x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch = math.degrees(math.asin(t2))
            t3 = +2.0 * (q.w * q.z + q.x * q.y)
            t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.degrees(math.atan2(t3, t4))
            return roll, pitch, yaw
            
        try:
            with open(filename, 'w') as f:
                f.write("=== SAM3 Final Poses ===\n\n")
                
                def write_tf(f, header_text, tfs):
                    f.write(header_text + "\n")
                    for t in tfs:
                        pos = t.transform.translation
                        rot = t.transform.rotation
                        r, p, y = quat_to_euler(rot)
                        f.write(f"Frame: {t.child_frame_id} | X: {pos.x:.4f} | Y: {pos.y:.4f} | Z: {pos.z:.4f} | "
                                f"oX: {rot.x:.4f} | oY: {rot.y:.4f} | oZ: {rot.z:.4f} | oW: {rot.w:.4f} | "
                                f"Roll: {r:.4f} | Pitch: {p:.4f} | Yaw: {y:.4f}\n")
                
                write_tf(f, "--- Landmarks ---", getattr(self, 'latest_landmark_tfs', []))
                f.write("\n")
                write_tf(f, "--- SAM3 Boxes ---", getattr(self, 'latest_box_tfs', []))
                f.write("\n")
                write_tf(f, "--- SAM3 Rotated Boxes ---", getattr(self, 'latest_rotated_tfs', []))
                    
            self.get_logger().info(f"Final poses successfully saved to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save final poses: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SAM3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_poses_to_file()
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()