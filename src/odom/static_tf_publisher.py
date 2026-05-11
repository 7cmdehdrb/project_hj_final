import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf_transformations # 쿼터니언 변환을 위해 필요할 수 있음

class StaticCameraTfPublisher(Node):
    def __init__(self):
        super().__init__('static_camera_tf_publisher')

        # StaticTransformBroadcaster 초기화
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # 변환 정보 설정
        self.publish_static_transform()

    def publish_static_transform(self):
        t = TransformStamped()

        # 타임스탬프와 프레임 ID 설정
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'tool0_controller' # 부모 프레임
        t.child_frame_id = 'camera_link'      # 자식 프레임

        # 좌표 이동 (Translation) - 미터(m) 단위
        t.transform.translation.x = 0.05
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # 회전 (Rotation) - 쿼터니언(Quaternion) 단위
        # Euler (0, 0, 0) -> Quaternion (0, 0, 0, 1)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # 데이터 발행
        self.tf_static_broadcaster.sendTransform(t)
        self.get_logger().info('Static transform from tool0_controller to camera_link published.')

def main():
    rclpy.init()
    node = StaticCameraTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()