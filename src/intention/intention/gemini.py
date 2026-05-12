import os
import tempfile
import subprocess
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class VirtualRobotSpawner(Node):
    def __init__(self):
        super().__init__("virtual_robot_spawner")

        # /robot_description은 Transient Local 정책으로 발행되므로 QoS 설정 필요
        qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.subscription = self.create_subscription(
            String, "/robot_description", self.listener_callback, qos_profile
        )

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.rsp_process = None

        self.get_logger().info("원본 '/robot_description'을 기다리는 중...")

    def listener_callback(self, msg):
        # 한 번만 URDF를 받아오면 되므로 이미 실행 중이면 무시
        if self.rsp_process is not None:
            return

        self.get_logger().info(
            "원본 URDF를 수신했습니다. 가상 로봇 모델을 생성합니다..."
        )

        original_urdf = msg.data
        virtual_urdf = self.modify_urdf(original_urdf, prefix="virtual_")

        # 수정된 URDF를 임시 파일로 저장 (robot_state_publisher에 전달하기 위함)
        fd, temp_path = tempfile.mkstemp(suffix=".urdf")
        with os.fdopen(fd, "w") as f:
            f.write(virtual_urdf)

        # 새로운 robot_state_publisher 서브프로세스로 실행
        cmd = [
            "ros2",
            "run",
            "robot_state_publisher",
            "robot_state_publisher",
            temp_path,
            "--ros-args",
            "-r",
            "robot_description:=/virtual_robot_description",  # 토픽명 리맵
            "-r",
            "__node:=virtual_robot_state_publisher",  # 노드명 리맵
        ]

        self.rsp_process = subprocess.Popen(cmd)
        self.get_logger().info(
            "가상 로봇의 robot_state_publisher가 성공적으로 실행되었습니다."
        )

        # 원래 로봇과 겹쳐보이지 않게 y축으로 1m 띄우는 Static TF 발행
        self.publish_static_tf()

        self.get_logger().info(
            "RViz에서 'RobotModel'을 추가하고 Topic을 '/virtual_robot_description'으로 설정하세요."
        )

    def modify_urdf(self, urdf_str, prefix="virtual_"):
        """URDF XML을 파싱하여 링크 이름에만 prefix를 붙입니다."""
        root = ET.fromstring(urdf_str)

        # 1. <link> 태그의 name 속성 변경
        for link in root.findall("link"):
            name = link.get("name")
            if name:
                link.set("name", prefix + name)

        # 2. <joint> 태그 내의 parent, child 링크 참조 변경
        # 주의: <joint name=".."> 자체는 원본과 동일하게 유지해야 /joint_states 와 동기화됨!
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            if parent is not None:
                p_name = parent.get("link")
                if p_name:
                    parent.set("link", prefix + p_name)

            child = joint.find("child")
            if child is not None:
                c_name = child.get("link")
                if c_name:
                    child.set("link", prefix + c_name)

        return ET.tostring(root, encoding="unicode")

    def publish_static_tf(self):
        """base_link와 virtual_base_link를 연결하는 TF 발행"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"  # 원본 로봇의 기준 프레임
        t.child_frame_id = "virtual_base_link"  # 가상 로봇의 기준 프레임

        # 원본 로봇과 안겹치게 위치 오프셋 (Y축 1m) -> 필요에 따라 0.0으로 변경
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        # 노드 종료 시 서브프로세스도 함께 정리
        if self.rsp_process:
            self.rsp_process.terminate()
            self.rsp_process.wait()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VirtualRobotSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
