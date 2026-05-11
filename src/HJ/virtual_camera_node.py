import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np
import cv2
from cv_bridge import CvBridge
import os

class VirtualCameraNode(Node):
    def __init__(self):
        super().__init__('virtual_camera_node')
        
        # QoS 설정 (Transient Local 유지)
        qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscription = self.create_subscription(
            PointCloud2, '/cloud_pcd', self.pcd_callback, qos_profile)
            
        self.img_pub = self.create_publisher(Image, '/virtual_camera/image', 10)
        self.xyz_pub = self.create_publisher(Image, '/virtual_camera/xyz_map', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/virtual_camera/camera_info', 10)
        self.bridge = CvBridge()
        
        # 1. 카메라 내부 인자 (해상도 및 화각)
        self.width, self.height = 640, 480
        self.fx, self.fy = 500.0, 500.0
        self.cx, self.cy = 320.0, 240.0
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ])
        
        # 최대 깊이 임계값 (이 깊이보다 먼 점은 이미지로 변환하지 않음, 필요에 따라 조절)
        self.max_depth = 1.2
                           
        # 2. 카메라 외부 인자 (위치 및 회전)
        # self.t = np.array([[0.25], [-0.5], [0.6]]) 
        self.t = np.array([[-0.5], [0.0], [0.35]]) 
        # self.R = np.array([
        #     [0.0,  0.0,  -1.0],
        #     [1.0,  0.0,  0.0],
        #     [0.0,  -1.0,  0.0]
        # ])
        self.R = np.array([
            [0.0,  0.0,  1.0],
            [-1.0,  0.0,  0.0],
            [0.0,  -1.0,  0.0]
        ])

        # self.R = np.array([
        #     [1.0,  0.0,  0.0],
        #     [0.0,  1.0,  0.0],
        #     [0.0,  0.0,  1.0]
        # ])

        # TF 브로드캐스터
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_static_tf()

        # 캐싱용 변수 및 타이머 (10Hz)
        self.latest_points = None
        self.latest_colors = None
        self.timer = self.create_timer(0.1, self.timer_callback)

        # -------------------------------
        # 이미지 저장 관련 설정
        # -------------------------------
        self.save_dir = 'saved_virtual_images'
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_image_enabled = True      # 저장 기능 on/off
        self.save_only_once = False         # True면 1장만 저장
        self.save_interval = 10             # 10프레임마다 1장 저장
        self.frame_count = 0
        self.image_saved_once = False

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

    def broadcast_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'virtual_camera_link'

        t.transform.translation.x = float(self.t[0][0])
        t.transform.translation.y = float(self.t[1][0])
        t.transform.translation.z = float(self.t[2][0])

        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(self.R)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def pcd_callback(self, msg):
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        pc_list = list(pc_data)
        if not pc_list:
            return
            
        self.latest_points = np.array([[p[0], p[1], p[2]] for p in pc_list]).T 
        
        colors = []
        for p in pc_list:
            rgb_val = int(p[3]) 
            r = (rgb_val >> 16) & 255
            g = (rgb_val >> 8) & 255
            b = rgb_val & 255
            colors.append([b, g, r])
        self.latest_colors = np.array(colors)

    def save_image(self, img, stamp):
        if not self.save_image_enabled:
            return

        if self.save_only_once and self.image_saved_once:
            return

        if not self.save_only_once:
            if self.frame_count % self.save_interval != 0:
                return

        sec = stamp.sec
        nanosec = stamp.nanosec
        filename = os.path.join(self.save_dir, f"virtual_camera_{sec}_{nanosec}.png")

        success = cv2.imwrite(filename, img)
        if success:
            self.get_logger().info(f"Saved image: {filename}")
            self.image_saved_once = True
        else:
            self.get_logger().warn(f"Failed to save image: {filename}")

    def timer_callback(self):
        if self.latest_points is None or self.latest_colors is None:
            return

        # 원본 Map 좌표 백업용
        map_points = self.latest_points
        
        # Map -> Camera 좌표계 변환
        P_cam = self.R.T @ (self.latest_points - self.t)
        
        # 렌즈 뒤쪽(Z<=0) 및 설정된 최대 깊이 이상인 점 필터링
        valid_z = (P_cam[2, :] > 0.01) & (P_cam[2, :] <= self.max_depth)
        P_cam = P_cam[:, valid_z]
        colors = self.latest_colors[valid_z]
        map_points = map_points[:, valid_z]
        
        # 깊이(Z) 정렬
        depths = P_cam[2, :]
        sort_indices = np.argsort(depths)[::-1] 
        P_cam = P_cam[:, sort_indices]
        colors = colors[sort_indices]
        map_points = map_points[:, sort_indices]

        # Camera -> 2D 픽셀 좌표계 변환
        P_pixel = self.K @ P_cam
        u = (P_pixel[0, :] / P_pixel[2, :]).astype(int)
        v = (P_pixel[1, :] / P_pixel[2, :]).astype(int)
        
        valid_uv = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        u = u[valid_uv]
        v = v[valid_uv]
        colors = colors[valid_uv]
        map_points = map_points[:, valid_uv]
        
        # 순수 원본 캔버스 렌더링
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[v, u] = colors
        
        # 3D Map 포즈 좌표 맵 생성
        xyz_map = np.full((self.height, self.width, 3), np.nan, dtype=np.float32)
        xyz_map[v, u] = map_points.T
        
        # 점 사이의 빈틈만 살짝 메움
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

        now = self.get_clock().now().to_msg()

        # 이미지 저장
        # self.frame_count += 1
        # self.save_image(img, now)

        # Image 토픽 발행
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = "virtual_camera_link"
        self.img_pub.publish(img_msg)

        # xyz_map 토픽 발행
        xyz_msg = self.bridge.cv2_to_imgmsg(xyz_map, encoding="32FC3")
        xyz_msg.header.stamp = now
        xyz_msg.header.frame_id = "map"
        self.xyz_pub.publish(xyz_msg)

        # CameraInfo 토픽 발행
        info_msg = CameraInfo()
        info_msg.header.stamp = now
        info_msg.header.frame_id = "virtual_camera_link"
        info_msg.width = self.width
        info_msg.height = self.height
        info_msg.k = self.K.flatten().tolist()
        
        P = np.zeros((3, 4))
        P[:3, :3] = self.K
        info_msg.p = P.flatten().tolist()
        
        self.info_pub.publish(info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VirtualCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()