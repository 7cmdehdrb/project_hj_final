#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_POSE_FILE = (
    "/home/irol/project_hj_final/src/object_estimation/resource/"
    "sam3_gravity_poses_0512.txt"
)


@dataclass(frozen=True)
class Sam3BoxPose:
    marker_id: int
    frame: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


class Sam3VirtualBoxPublisher(Node):
    def __init__(self) -> None:
        super().__init__("sam3_virtual_box_publisher")

        self.declare_parameter("pose_file", DEFAULT_POSE_FILE)
        self.declare_parameter("topic", "/virtual_box")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("scale_x", 0.01)
        self.declare_parameter("scale_y", 0.1)
        self.declare_parameter("scale_z", 0.1)

        self.pose_file = Path(self.get_string_parameter("pose_file"))
        self.topic = self.get_string_parameter("topic")
        self.frame_id = self.get_string_parameter("frame_id")
        self.scale = Vector3(
            x=self.get_double_parameter("scale_x"),
            y=self.get_double_parameter("scale_y"),
            z=self.get_double_parameter("scale_z"),
        )

        publish_rate_hz = self.get_double_parameter("publish_rate_hz")
        if publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be greater than zero")

        self.box_poses = self.load_sam3_box_poses(self.pose_file)
        self.publisher = self.create_publisher(MarkerArray, self.topic, 10)
        timer_period = 1.0 / publish_rate_hz
        self.timer = self.create_timer(timer_period, self.publish_markers)

        self.get_logger().info(
            f"Publishing {len(self.box_poses)} SAM3 boxes from "
            f"{self.pose_file} to {self.topic} in frame "
            f"'{self.frame_id}' at {publish_rate_hz:.1f} Hz"
        )

    def get_string_parameter(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def get_double_parameter(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def load_sam3_box_poses(self, pose_file: Path) -> List[Sam3BoxPose]:
        if not pose_file.exists():
            raise FileNotFoundError(f"Pose file does not exist: {pose_file}")

        lines = pose_file.read_text(encoding="utf-8").splitlines()
        in_sam3_boxes = False
        box_poses: List[Sam3BoxPose] = []

        for line in lines:
            stripped = line.strip()
            if stripped == "--- SAM3 Boxes ---":
                in_sam3_boxes = True
                continue

            if in_sam3_boxes and stripped.startswith("---"):
                break

            if not in_sam3_boxes or not stripped:
                continue

            marker_id = len(box_poses) + 1
            box_poses.append(self.parse_box_pose_line(stripped, marker_id))

        if len(box_poses) != 3:
            raise ValueError(
                "Expected 3 poses in '--- SAM3 Boxes ---', " f"found {len(box_poses)}"
            )

        return box_poses

    def parse_box_pose_line(self, line: str, marker_id: int) -> Sam3BoxPose:
        fields = {}
        for part in line.split("|"):
            key_value = part.strip().split(":", 1)
            if len(key_value) != 2:
                continue
            key, value = key_value
            fields[key.strip()] = value.strip()

        required_fields = ["Frame", "X", "Y", "Z", "oX", "oY", "oZ", "oW"]
        missing = [field for field in required_fields if field not in fields]
        if missing:
            raise ValueError(f"Missing fields {missing} in pose line: {line}")

        frame = fields["Frame"]
        match = re.search(r"(\d+)$", frame)
        parsed_marker_id = int(match.group(1)) if match else marker_id

        return Sam3BoxPose(
            marker_id=parsed_marker_id,
            frame=frame,
            x=float(fields["X"]),
            y=float(fields["Y"]),
            z=float(fields["Z"]),
            qx=float(fields["oX"]),
            qy=float(fields["oY"]),
            qz=float(fields["oZ"]),
            qw=float(fields["oW"]),
        )

    def publish_markers(self) -> None:
        now = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        for box_pose in self.box_poses:
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = now
            marker.ns = "sam3_virtual_boxes"
            marker.id = box_pose.marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = box_pose.x
            marker.pose.position.y = box_pose.y
            marker.pose.position.z = box_pose.z
            marker.pose.orientation.x = box_pose.qx
            marker.pose.orientation.y = box_pose.qy
            marker.pose.orientation.z = box_pose.qz
            marker.pose.orientation.w = box_pose.qw

            marker.scale = self.scale
            marker.color = ColorRGBA(r=0.0, g=0.7, b=1.0, a=0.45)
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 0

            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Sam3VirtualBoxPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
