#!/usr/bin/env python3

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

INPUT_TOPIC = "/robot_description"
OUTPUT_TOPIC = "/virtual_robot_description"
JOINT_STATE_INPUT_TOPIC = "/joint_states"
JOINT_STATE_OUTPUT_TOPIC = "/virtual_joint_states"
VIRTUAL_BASE_LINK = "virtual_base_link"
ROBOT_BASE_LINK = "base_link"
VIRTUAL_PREFIX = "virtual_"


def make_virtual_base_urdf(
    urdf_text: str,
    virtual_base_link: str = VIRTUAL_BASE_LINK,
    robot_base_link: str = ROBOT_BASE_LINK,
) -> str:
    virtual_urdf, _ = make_virtual_base_urdf_and_joint_map(
        urdf_text,
        virtual_base_link,
        robot_base_link,
    )
    return virtual_urdf


def make_virtual_base_urdf_and_joint_map(
    urdf_text: str,
    virtual_base_link: str = VIRTUAL_BASE_LINK,
    robot_base_link: str = ROBOT_BASE_LINK,
) -> tuple[str, dict[str, str]]:
    root = ET.fromstring(urdf_text)
    virtual_root = copy.deepcopy(root)
    virtual_root.set("name", f"{root.get('name', 'robot')}_virtual")

    link_name_map = make_link_name_map(
        virtual_root,
        robot_base_link,
        virtual_base_link,
    )
    joint_name_map = make_joint_name_map(virtual_root)

    rename_urdf_references(virtual_root, link_name_map, joint_name_map)

    return ET.tostring(virtual_root, encoding="unicode"), joint_name_map


def make_link_name_map(
    root: ET.Element,
    robot_base_link: str,
    virtual_base_link: str,
) -> dict[str, str]:
    link_name_map = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        if link_name is None:
            continue

        if link_name == robot_base_link:
            link_name_map[link_name] = virtual_base_link
        else:
            link_name_map[link_name] = virtual_name(link_name)

    return link_name_map


def make_joint_name_map(root: ET.Element) -> dict[str, str]:
    joint_name_map = {}
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        if joint_name is not None:
            joint_name_map[joint_name] = virtual_name(joint_name)

    return joint_name_map


def rename_urdf_references(
    root: ET.Element,
    link_name_map: dict[str, str],
    joint_name_map: dict[str, str],
) -> None:
    for element in root.iter():
        element_tag = tag_without_namespace(element.tag)

        if element_tag == "link":
            update_attribute(element, "name", link_name_map)

        if element_tag == "joint":
            update_attribute(element, "name", joint_name_map)

        update_attribute(element, "link", link_name_map)
        update_attribute(element, "reference", link_name_map)
        update_attribute(element, "frame", link_name_map)


def update_attribute(
    element: ET.Element,
    attribute_name: str,
    name_map: dict[str, str],
) -> None:
    value = element.get(attribute_name)
    if value in name_map:
        element.set(attribute_name, name_map[value])


def tag_without_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def virtual_name(name: str) -> str:
    if name.startswith(VIRTUAL_PREFIX):
        return name

    return f"{VIRTUAL_PREFIX}{name}"


class VirtualRobotDescriptionPublisher(Node):
    def __init__(self) -> None:
        super().__init__(
            "virtual_robot_description_publisher",
            enable_rosout=False,
            start_parameter_services=False,
        )
        self.disable_parameter_event_publisher()

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.publisher = self.create_publisher(String, OUTPUT_TOPIC, qos)
        self.joint_state_publisher = self.create_publisher(
            JointState,
            JOINT_STATE_OUTPUT_TOPIC,
            10,
        )
        self.subscription = self.create_subscription(
            String,
            INPUT_TOPIC,
            self.robot_description_callback,
            qos,
        )
        self.joint_state_subscription = self.create_subscription(
            JointState,
            JOINT_STATE_INPUT_TOPIC,
            self.joint_state_callback,
            10,
        )
        self.last_published_data = None
        self.joint_name_map = {}

    def disable_parameter_event_publisher(self) -> None:
        parameter_event_publisher = getattr(
            self,
            "_parameter_event_publisher",
            None,
        )
        if parameter_event_publisher is not None:
            self.destroy_publisher(parameter_event_publisher)
            self._parameter_event_publisher = None

    def robot_description_callback(self, msg: String) -> None:
        result = make_virtual_base_urdf_and_joint_map(msg.data)
        virtual_description, joint_name_map = result
        self.joint_name_map = joint_name_map

        if virtual_description == self.last_published_data:
            return

        self.publisher.publish(String(data=virtual_description))
        self.last_published_data = virtual_description

    def joint_state_callback(self, msg: JointState) -> None:
        if not self.joint_name_map:
            return

        virtual_joint_state = JointState()
        virtual_joint_state.header = msg.header

        for index, joint_name in enumerate(msg.name):
            virtual_joint_name = self.joint_name_map.get(joint_name)
            if virtual_joint_name is None:
                continue

            virtual_joint_state.name.append(virtual_joint_name)
            append_if_available(
                virtual_joint_state.position,
                msg.position,
                index,
            )
            append_if_available(
                virtual_joint_state.velocity,
                msg.velocity,
                index,
            )
            append_if_available(virtual_joint_state.effort, msg.effort, index)

        if virtual_joint_state.name:
            self.joint_state_publisher.publish(virtual_joint_state)


def append_if_available(
    output_values,
    input_values,
    index: int,
) -> None:
    if index < len(input_values):
        output_values.append(input_values[index])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VirtualRobotDescriptionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
