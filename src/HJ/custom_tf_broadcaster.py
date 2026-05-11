import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math
import numpy as np

class CustomTFBroadcaster(Node):
    def __init__(self):
        super().__init__('custom_tf_broadcaster')
        
        # TF 브로드캐스터 초기화
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 브로드캐스트 주기 설정 (예상: 10Hz)
        self.timer = self.create_timer(0.1, self.broadcast_tf)
        
        # ==========================================
        # 사용자가 설정할 수 있는 TF 위치 및 회전 값 세팅
        # ==========================================
        self.parent_frame = 'map'
        self.child_frame = 'my_custom_frame' # 원하는 자식 프레임 이름으로 변경하세요
        
        # Translation (x, y, z 좌표) [단위: meter]
        self.x = -0.825
        self.y = -0.104
        self.z = 0.320
        
        # Rotation (Roll, Pitch, Yaw 각도) [단위: degree]
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 90.0

        self.get_logger().info(f"Custom TF Broadcaster Initialized.")
        self.get_logger().info(f"Publishing TF: [{self.parent_frame}] -> [{self.child_frame}]")

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        오일러 각도(라디안)를 쿼터니언(x, y, z, w)으로 변환
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return qx, qy, qz, qw

    def broadcast_tf(self):
        t = TransformStamped()
        
        # 현재 시간과 프레임 정보 설정
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        
        # 위치 설정
        t.transform.translation.x = float(self.x)
        t.transform.translation.y = float(self.y)
        t.transform.translation.z = float(self.z)
        
        # 각도를 라디안으로 변환
        roll_rad = math.radians(self.roll_deg)
        pitch_rad = math.radians(self.pitch_deg)
        yaw_rad = math.radians(self.yaw_deg)
        
        # 오일러 -> 쿼터니언 변환 후 회전 설정
        qx, qy, qz, qw = self.euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
        
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        # TF 브로드캐스트
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = CustomTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
