import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry  # 추가된 부분
from tf2_ros import TransformBroadcaster
import math

class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher_node')
        
        # 1. 토픽 발행 설정 (RTAB-Map이 기다리는 토픽)
        self.odom_pub = self.create_publisher(Odometry, '/your_own_odom_topic', 10)
        
        # 2. TF 발행 설정
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.timer = self.create_timer(1/30.0, self.timer_callback)
        self.x, self.y, self.theta = 0.0, 0.0, 0.0

    def timer_callback(self):
        # 위치 업데이트 로직 (예시)
        current_time = self.get_clock().now().to_msg()
        self.x += 0.01 
        q = self.euler_to_quaternion(0, 0, self.theta)

        # --- [A] TF 발행 (odom -> camera_link) ---
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'odom'
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = self.x
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_broadcaster.sendTransform(t)

        # --- [B] Odometry 토픽 발행 (/your_own_odom_topic) ---
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'camera_link'
        
        # 위치 정보
        odom.pose.pose.position.x = self.x
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = q
        
        self.odom_pub.publish(odom)

    def euler_to_quaternion(self, roll, pitch, yaw):
        # (이전 코드와 동일한 변환 로직)
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()