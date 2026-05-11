import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters

import cv2
import numpy as np
import math

from ultralytics import YOLO

class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        
        # Initialize YOLO predictor
        yolo_model_path = "/home/irol/ros2_ws/src/HJ/yolo/best.pt"
        self.get_logger().info(f"Loading YOLO model from {yolo_model_path} ...")
        self.model = YOLO(yolo_model_path)
        
        # RViz Maker Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/yolo/box_markers', 10)
        
        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Configure subscriptions with message_filters
        self.image_sub = message_filters.Subscriber(self, Image, '/virtual_camera/image')
        self.xyz_sub = message_filters.Subscriber(self, Image, '/virtual_camera/xyz_map')
        
        # ApproximateTimeSynchronizer allows small delay between topics
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.xyz_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        
        self.get_logger().info("YOLO Node Initialized. Waiting for synchronized images...")

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
            
        # Perform YOLO inference
        results = self.model(cv_image, verbose=False)
        boxes = results[0].boxes
        
        if boxes is not None and len(boxes) > 0:
            boxes_np = boxes.xyxy.cpu().numpy()
            
            marker_array = MarkerArray()
            # Delete all previous markers to prevent ghost boxes
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            
            # Print bounding box & Map Pose
            for i, box in enumerate(boxes_np):
                # 2D Bounding Box 좌표 추출 (이미지 경계선 처리)
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(xyz_map.shape[1], x2), min(xyz_map.shape[0], y2)
                
                # Bounding box 영역의 3D 포인트들 추출
                points_3d = xyz_map[y1:y2, x1:x2].reshape(-1, 3)
                
                # Filter out NaNs (where depth/xyz was invalid)
                valid_points = points_3d[~np.isnan(points_3d).any(axis=1)]
                
                if len(valid_points) > 3: # PCA를 위해 최소한의 점이 필요함
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
                    # 법선 벡터가 항상 카메라 쪽(+X 방향)을 보도록 부호 일치 방지
                    if uz[0] < 0:
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
                    marker.ns = "yolo_boxes"
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
                    t.child_frame_id = f"yolo_box_{i+1}"
                    
                    t.transform.translation.x = float(centroid[0])
                    t.transform.translation.y = float(centroid[1])
                    t.transform.translation.z = float(centroid[2])
                    
                    # 주성분 분석(PCA)으로 구한 orientation 적용
                    t.transform.rotation.x = qx
                    t.transform.rotation.y = qy
                    t.transform.rotation.z = qz
                    t.transform.rotation.w = qw
                    self.tf_broadcaster.sendTransform(t)
                    
                else:
                    self.get_logger().warn(f"Detected Box {i+1}: No valid 3D points found inside the mask.")
            
            if len(marker_array.markers) > 1: # only DELETEALL is not enough
                self.marker_pub.publish(marker_array)
            
            # Visualize results (optional)
            # result_img = results[0].plot()
            # cv2.imshow("YOLO Bounding Box", result_img)
            # cv2.waitKey(1)
            
        else:
            self.get_logger().info("No 'box' detected.")
            # cv2.imshow("YOLO Bounding Box", cv_image)
            # cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YOLONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
