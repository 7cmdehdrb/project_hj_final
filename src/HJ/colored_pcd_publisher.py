import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, DurabilityPolicy
import open3d as o3d
import numpy as np

class ColoredPCDPublisher(Node):
    def __init__(self):
        super().__init__('colored_pcd_publisher')
        
        # QoS 설정 (Latching)
        qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher_ = self.create_publisher(PointCloud2, '/cloud_pcd', qos_profile)
        
        # 1. PCD 파일 로드
        # pcd_path = '/home/irol/ros2_ws/src/HJ/pcd_file/240910_noiseX.pcd'
        pcd_path = '/home/irol/ros2_ws/src/HJ/pcd_file/map0511.pcd'
        self.get_logger().info(f'PCD 로드 중: {pcd_path}')
        pcd = o3d.io.read_point_cloud(pcd_path)

        # 복셀 다운샘플링 (너무 렉이 걸리면 0.005 등의 값 적용, 지금은 주석 처리)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) # Open3D는 색상을 0.0 ~ 1.0 사이로 불러옴
        
        # 색상 데이터 존재 여부 확인
        if len(colors) == 0:
            self.get_logger().error('PCD 파일에 색상(RGB) 데이터가 없습니다!')
            return
            
        self.get_logger().info('XYZRGB 데이터 초고속 변환 중...')
        
        # 2. 색상 데이터 변환 ([0, 1] -> [0, 255])
        r = np.asarray(colors[:, 0] * 255.0, dtype=np.uint32)
        g = np.asarray(colors[:, 1] * 255.0, dtype=np.uint32)
        b = np.asarray(colors[:, 2] * 255.0, dtype=np.uint32)
        
        # 비트 연산으로 R, G, B를 하나의 32비트 정수로 압축 (ROS2 호환 포맷)
        rgb = (r << 16) | (g << 8) | b
        
        # 3. Numpy 구조체 배열 생성 (파이썬 for문 없이 한 번에 메모리 할당)
        ros_dtype = np.dtype([
            ('x', np.float32), 
            ('y', np.float32), 
            ('z', np.float32), 
            ('rgb', np.uint32)
        ])
        cloud_data = np.empty(len(points), dtype=ros_dtype)
        cloud_data['x'] = points[:, 0]
        cloud_data['y'] = points[:, 1]
        cloud_data['z'] = points[:, 2]
        cloud_data['rgb'] = rgb

        # 4. PointCloud2 메시지 조립
        msg = PointCloud2()
        msg.header = Header()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = False
        msg.is_bigendian = False
        
        # X, Y, Z, RGB 필드 메모리 오프셋 정의
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16 # 하나의 포인트가 차지하는 바이트 수 (4바이트 * 4개)
        msg.row_step = msg.point_step * len(points)
        
        # Numpy 배열을 그대로 바이트로 쏟아부음 (가장 빠른 방법)
        msg.data = cloud_data.tobytes() 

        self.publisher_.publish(msg)
        self.get_logger().info(f'색상이 포함된 {len(points)}개의 포인트 발행 완료!')

def main(args=None):
    rclpy.init(args=args)
    node = ColoredPCDPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()