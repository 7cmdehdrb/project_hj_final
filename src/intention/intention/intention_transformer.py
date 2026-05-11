#!/usr/bin/env python3
"""
ROS2 single-file implementation for real/virtual intention estimation and
map -> virtual_base_link TF publication.

This file replaces the previous ROS1 nodes:
- real_intention_gaussian_v3_with_pdf.py
- virtual_intention_gaussian_v3_with_pdf.py
- pose_transformer.py

Core changes:
- ROS1 rospy/tf/custom_msgs dependencies are removed.
- Box inputs are visualization_msgs/msg/MarkerArray.
- Best boxes are published as one MarkerArray on /best_box/selected.
- tool0 twist is computed by differentiating TF poses.
- map -> virtual_base_link is published with full pose alignment by default.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import multivariate_normal

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Pose, TransformStamped, Vector3
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from rotutils import euler_from_quaternion, euler_from_rotation_matrix,  quaternion_from_euler, quaternion_from_rotation_matrix, rot

# =============================================================================
# Hard-coded configuration section
# =============================================================================
# Do not use ROS2 declare_parameter. Edit these variables directly.

NODE_NAME = "intention_transform_node"

REAL_BOX_TOPIC = "/real_box/eb"
VIRTUAL_BOX_TOPIC = "/virtual_box"
BEST_BOX_SELECTED_TOPIC = "/best_box/selected"

JOINT_STATE_TOPIC = "/joint_states"
JOY_TOPIC = "/controller/right/joy"
RESET_BUTTON_INDEX = 2

MAP_FRAME = "map"
BASE_FRAME = "base_link"
TOOL_FRAME = "tool0"
VIRTUAL_BASE_FRAME = "virtual_base_link"

# If True, box orientations are ignored during final TF alignment.
# False: T_map_virtual_base = T_map_virtual_box @ inv(T_base_real_box)
# True : p_map_virtual_base = p_map_virtual_box - p_base_real_box, q = identity
IGNORE_ORIENTATION = False

# Intention update and TF publication rates.
INTENTION_RATE_HZ = 10.0
TF_RATE_HZ = 30.0

# Gaussian intention parameters.
MIN_INTERSECTION_SAMPLES = 3
MANIPULATOR_LENGTH_R = 1.125
GAIN_DISTANCE_A = 1.0
MAX_INTERSECTION_T = 30.0

# Plane parameters from the previous implementation.
# The same coefficients are used in each intention instance's target frame.
PLANE_NORMAL = np.array([0.99887537, 0.0465492, 0.00900962], dtype=float)
PLANE_D = -0.8969238154115642

# TF lookup timeout.
TF_LOOKUP_TIMEOUT_SEC = 0.05

# Best-box visualization settings.
BEST_BOX_REAL_MARKER_ID = 0
BEST_BOX_VIRTUAL_MARKER_ID = 1
BEST_BOX_REAL_NS = "real_best_box"
BEST_BOX_VIRTUAL_NS = "virtual_best_box"
BEST_BOX_SCALE_FALLBACK = Vector3(x=0.03, y=0.25, z=0.15)


# =============================================================================
# Math utilities
# =============================================================================


def stamp_to_sec(stamp: TimeMsg) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def now_msg(node: Node) -> TimeMsg:
    return node.get_clock().now().to_msg()


def normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = normalize_quaternion_xyzw(q)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_xyzw(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]."""
    m = np.asarray(rot, dtype=float)
    trace = np.trace(m)

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return normalize_quaternion_xyzw(np.array([x, y, z, w], dtype=float))


def pose_to_matrix(pose: Pose, ignore_orientation: bool = False) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, 3] = np.array(
        [pose.position.x, pose.position.y, pose.position.z], dtype=float
    )

    if not ignore_orientation:
        q = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=float,
        )
        mat[:3, :3] = quaternion_xyzw_to_matrix(q)

    return mat


def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    t = transform.transform.translation
    q = transform.transform.rotation
    mat[:3, 3] = np.array([t.x, t.y, t.z], dtype=float)
    mat[:3, :3] = quaternion_xyzw_to_matrix(np.array([q.x, q.y, q.z, q.w]))
    return mat


def matrix_to_transform_stamped(
    mat: np.ndarray,
    parent_frame: str,
    child_frame: str,
    stamp: TimeMsg,
) -> TransformStamped:
    mat = np.asarray(mat, dtype=float)
    q = matrix_to_quaternion_xyzw(mat[:3, :3])

    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = float(mat[0, 3])
    msg.transform.translation.y = float(mat[1, 3])
    msg.transform.translation.z = float(mat[2, 3])
    msg.transform.rotation.x = float(q[0])
    msg.transform.rotation.y = float(q[1])
    msg.transform.rotation.z = float(q[2])
    msg.transform.rotation.w = float(q[3])
    return msg


def copy_marker_for_best_box(
    src: Marker,
    marker_id: int,
    namespace: str,
    stamp: TimeMsg,
    color: ColorRGBA,
) -> Marker:
    marker = copy.deepcopy(src)
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.action = Marker.ADD
    marker.color = color

    if marker.scale.x == 0.0 and marker.scale.y == 0.0 and marker.scale.z == 0.0:
        marker.scale = BEST_BOX_SCALE_FALLBACK

    return marker


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class BoxRecord:
    id: int
    pose: Pose
    frame_id: str
    marker: Marker


@dataclass
class BestBox:
    box: BoxRecord
    possibility: float


@dataclass
class ToolState:
    position: np.ndarray
    linear_velocity: np.ndarray
    stamp_sec: float


# =============================================================================
# Core classes
# =============================================================================


class Plane:
    def __init__(self, n: np.ndarray, d: float):
        if not isinstance(n, np.ndarray):
            raise ValueError("n must be a numpy array")
        if n.shape != (3,):
            raise ValueError("n must have shape (3,)")
        if np.linalg.norm(n) < 1e-12:
            raise ValueError("plane normal vector must be non-zero")

        self.n = n.astype(float)
        self.d = float(d)

    def distance(self, p: np.ndarray) -> float:
        return float(abs(np.dot(self.n, p) + self.d) / np.linalg.norm(self.n))

    def project_to_plane(
        self, p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        point_on_plane = p - (np.dot(self.n, p) + self.d) * self.n

        u = np.array([-self.n[1], self.n[0], 0.0], dtype=float)
        if np.allclose(u, 0.0):
            u = np.array([0.0, -self.n[2], self.n[1]], dtype=float)
        u = u / np.linalg.norm(u)
        v = np.cross(self.n, u)
        v = v / np.linalg.norm(v)

        x = np.dot(point_on_plane, u)
        y = np.dot(point_on_plane, v)
        return np.array([x, y], dtype=float), u, v

    def transform_covariance(
        self, cov_3d: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        transform = np.stack((u, v), axis=0)
        return transform @ cov_3d @ transform.T

    def transform_to_2d(
        self, position: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if position.shape != (3,) or covariance.shape != (3, 3):
            raise ValueError("position must be (3,), covariance must be (3, 3)")

        pos_2d, u, v = self.project_to_plane(position)
        cov_2d = self.transform_covariance(covariance, u, v)
        return pos_2d, cov_2d


class BoxManager:
    def __init__(self, node: Node, topic: str, expected_frame: str, name: str):
        self.node = node
        self.topic = topic
        self.expected_frame = expected_frame
        self.name = name
        self.boxes: Dict[int, BoxRecord] = {}

        self.sub = node.create_subscription(
            MarkerArray,
            topic,
            self.callback,
            10,
        )

    def callback(self, msg: MarkerArray) -> None:
        # The incoming MarkerArray is a full fresh observation, but missing markers
        # are intentionally not deleted because transient perception dropout is possible.
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue
            if marker.action == Marker.DELETE:
                continue

            frame_id = marker.header.frame_id or self.expected_frame
            if frame_id != self.expected_frame:
                self.node.get_logger().warn(
                    f"[{self.name}] marker frame_id='{frame_id}' differs from "
                    f"expected_frame='{self.expected_frame}'. The marker is still accepted."
                )

            self.boxes[int(marker.id)] = BoxRecord(
                id=int(marker.id),
                pose=copy.deepcopy(marker.pose),
                frame_id=frame_id,
                marker=copy.deepcopy(marker),
            )

    def get_boxes(self) -> Dict[int, BoxRecord]:
        return self.boxes


class Robot:
    def __init__(self, node: Node, tf_buffer: Buffer):
        self.node = node
        self.tf_buffer = tf_buffer
        self.latest_joint_state: Optional[JointState] = None
        self.previous_tool_states: Dict[str, ToolState] = {}

        self.joint_state_sub = node.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self.joint_state_callback,
            10,
        )

    def joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def get_joint_state(self) -> Optional[JointState]:
        return self.latest_joint_state

    def lookup_tool_position(
        self, target_frame: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                TOOL_FRAME,
                Time(),
                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC),
            )
        except TransformException as exc:
            self.node.get_logger().warn(
                f"Failed to lookup TF {target_frame} -> {TOOL_FRAME}: {exc}"
            )
            return None

        t = transform.transform.translation
        position = np.array([t.x, t.y, t.z], dtype=float)

        stamp_sec = stamp_to_sec(transform.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self.node.get_clock().now().nanoseconds * 1e-9

        return position, stamp_sec

    def get_tool_state(self, target_frame: str) -> Optional[ToolState]:
        lookup = self.lookup_tool_position(target_frame)
        if lookup is None:
            return None

        position, stamp_sec = lookup
        prev = self.previous_tool_states.get(target_frame)

        if prev is None:
            velocity = np.zeros(3, dtype=float)
        else:
            dt = stamp_sec - prev.stamp_sec
            if dt <= 1e-6:
                velocity = prev.linear_velocity.copy()
            else:
                velocity = (position - prev.position) / dt

        state = ToolState(
            position=position, linear_velocity=velocity, stamp_sec=stamp_sec
        )
        self.previous_tool_states[target_frame] = state
        return state


class Intention:
    class IntersectionMethod:
        @staticmethod
        def get_gain_distance(d: float, a: float = 1.0) -> float:
            d = max(float(d), 1e-6)
            return 1.0 + (1.0 / (a * d))

        @staticmethod
        def get_gain_theta(
            theta: float, trust_threshold: float, distrust_threshold: float
        ) -> float:
            theta = abs(float(theta))
            trust_threshold = float(trust_threshold)
            distrust_threshold = float(distrust_threshold)

            if theta < trust_threshold:
                return 1.0
            if distrust_threshold <= trust_threshold:
                return 0.0
            if trust_threshold <= theta < distrust_threshold:
                return float(
                    math.cos(
                        (math.pi / (2.0 * (distrust_threshold - trust_threshold)))
                        * (theta - trust_threshold)
                    )
                )
            return 0.0

        @staticmethod
        def get_theta(v: np.ndarray, plane: Plane) -> float:
            if np.linalg.norm(v) < 1e-12:
                return 0.0

            numerator = float(np.dot(v, plane.n))
            denominator = float(np.linalg.norm(v) * np.linalg.norm(plane.n))
            cos_value = float(np.clip(numerator / denominator, -1.0, 1.0))
            return float(np.arccos(cos_value) % (2.0 * np.pi))

        @staticmethod
        def get_intersection_point(
            p: np.ndarray,
            v: np.ndarray,
            plane: Plane,
        ) -> Tuple[bool, np.ndarray]:
            denominator = float(np.dot(plane.n, v))
            numerator = float(plane.n @ p + plane.d)

            if abs(denominator) < 1e-12:
                return True, np.array([np.nan, np.nan, np.nan], dtype=float)

            t = -numerator / denominator
            return (0.0 < t < MAX_INTERSECTION_T), p + (t * v)

        @staticmethod
        def get_weight_mean_and_covariance(
            data: np.ndarray,
            weights: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            if len(data) < MIN_INTERSECTION_SAMPLES:
                raise ValueError("not enough intersection samples")

            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                raise ValueError("sum of weights is too small")

            weighted_mean = np.average(data, axis=0, weights=weights)
            centered_data = data - weighted_mean
            weighted_covariance = np.cov(centered_data.T, aweights=weights, bias=True)

            # Numerical stabilization for nearly singular covariance.
            weighted_covariance = np.asarray(weighted_covariance, dtype=float)
            weighted_covariance += np.eye(3) * 1e-9
            return weighted_mean, weighted_covariance

        @staticmethod
        def get_trust_theta(p: np.ndarray, plane: Plane) -> Tuple[float, float]:
            radius = MANIPULATOR_LENGTH_R
            base = np.array([0.0, 0.0, 0.0], dtype=float)

            D = plane.distance(base)
            d = max(plane.distance(p), 1e-6)

            cls = Intention.IntersectionMethod
            check_eef, eef_intersection = cls.get_intersection_point(p, plane.n, plane)
            check_base, base_intersection = cls.get_intersection_point(
                base, plane.n, plane
            )

            if (
                check_eef
                and check_base
                and not np.isnan(eef_intersection).any()
                and not np.isnan(base_intersection).any()
            ):
                ratio = float(np.clip(D / radius, -1.0, 1.0))
                theta_D = float(np.arccos(ratio))

                # Preserve the previous implementation's use of [1:] components.
                lateral_distance = float(
                    np.linalg.norm(base_intersection[1:] - eef_intersection[1:])
                )
                radius_on_plane = abs(D * math.tan(theta_D))

                theta_d_max = math.atan((radius_on_plane + lateral_distance) / d)
                theta_d_min = math.atan((radius_on_plane - lateral_distance) / d)
                return float(theta_d_min), float(theta_d_max)

            return 0.0, 0.0

    def __init__(
        self,
        node: Node,
        name: str,
        robot: Robot,
        box_topic: str,
        box_frame: str,
        eef_target_frame: str,
        plane: Plane,
    ):
        self.node = node
        self.name = name
        self.robot = robot
        self.box_manager = BoxManager(node, box_topic, box_frame, name)
        self.eef_target_frame = eef_target_frame
        self.plane = plane

        self.mean_2d = np.zeros(2, dtype=float)
        self.cov_2d = np.eye(2, dtype=float)
        self.has_distribution = False

        self.intersections = np.empty((0, 3), dtype=float)
        self.gains = np.empty(0, dtype=float)
        self.distances = np.empty(0, dtype=float)
        self.reverse_distances = np.empty(0, dtype=float)

        self.last_direction_forward = True
        self.best_box: Optional[BestBox] = None

    def reset(self) -> None:
        self.mean_2d = np.zeros(2, dtype=float)
        self.cov_2d = np.eye(2, dtype=float)
        self.has_distribution = False

        self.intersections = np.empty((0, 3), dtype=float)
        self.gains = np.empty(0, dtype=float)
        self.distances = np.empty(0, dtype=float)
        self.reverse_distances = np.empty(0, dtype=float)

        self.last_direction_forward = True
        self.best_box = None

        self.node.get_logger().info(f"[{self.name}] intention state reset")

    def update_distribution(self) -> bool:
        tool_state = self.robot.get_tool_state(self.eef_target_frame)
        if tool_state is None:
            return False

        p = tool_state.position
        v = tool_state.linear_velocity

        distance_between_plane = self.plane.distance(p)
        theta_between_plane = self.IntersectionMethod.get_theta(v, self.plane)

        forward, intersection = self.IntersectionMethod.get_intersection_point(
            p, v, self.plane
        )
        trust_theta, distrust_theta = self.IntersectionMethod.get_trust_theta(
            p, self.plane
        )

        gain_distance = self.IntersectionMethod.get_gain_distance(
            distance_between_plane, GAIN_DISTANCE_A
        )
        gain_theta = self.IntersectionMethod.get_gain_theta(
            theta_between_plane,
            trust_theta,
            distrust_theta,
        )
        gain_total = gain_distance * gain_theta

        if np.isnan(intersection).any():
            return False

        if forward:
            if self.last_direction_forward != forward:
                if len(self.reverse_distances) > 2:
                    max_reverse_dist = np.max(self.reverse_distances)
                    mask = self.distances > max_reverse_dist
                    self.intersections = self.intersections[mask]
                    self.gains = self.gains[mask]
                    self.distances = self.distances[mask]

                self.reverse_distances = np.empty(0, dtype=float)

            self.intersections = np.vstack([self.intersections, intersection])
            self.gains = np.append(self.gains, gain_total)
            self.distances = np.append(self.distances, distance_between_plane)
        else:
            self.reverse_distances = np.append(
                self.reverse_distances, distance_between_plane
            )

        self.last_direction_forward = forward

        if len(self.intersections) < MIN_INTERSECTION_SAMPLES:
            return False

        try:
            mean_3d, cov_3d = self.IntersectionMethod.get_weight_mean_and_covariance(
                self.intersections,
                self.gains,
            )
            mean_2d, cov_2d = self.plane.transform_to_2d(mean_3d, cov_3d)
        except Exception as exc:
            self.node.get_logger().warn(
                f"[{self.name}] distribution update failed: {exc}"
            )
            return False

        self.mean_2d = mean_2d
        self.cov_2d = cov_2d + np.eye(2) * 1e-9
        self.has_distribution = True
        return True

    def update_best_box(self) -> Optional[BestBox]:
        if not self.has_distribution:
            return self.best_box

        boxes = self.box_manager.get_boxes()
        if not boxes:
            return self.best_box

        try:
            rv = multivariate_normal(
                mean=self.mean_2d,
                cov=self.cov_2d,
                allow_singular=True,
            )
            max_reference_pdf = float(rv.pdf(self.mean_2d))
        except Exception as exc:
            self.node.get_logger().warn(
                f"[{self.name}] Gaussian creation failed: {exc}"
            )
            return self.best_box

        if max_reference_pdf <= 1e-12:
            return self.best_box

        max_pdf = -1.0
        selected_box: Optional[BoxRecord] = None

        for box in boxes.values():
            box_position = np.array(
                [box.pose.position.x, box.pose.position.y, box.pose.position.z],
                dtype=float,
            )
            box_position_2d, _, _ = self.plane.project_to_plane(box_position)
            pdf = float(rv.pdf(box_position_2d))

            if pdf > max_pdf:
                max_pdf = pdf
                selected_box = box

        if selected_box is None:
            return self.best_box

        possibility = max_pdf / max_reference_pdf
        self.best_box = BestBox(box=selected_box, possibility=float(possibility))

        self.node.get_logger().info(
            f"[{self.name}] boxes={len(boxes)}, best_id={selected_box.id}, "
            f"possibility={possibility:.4f}"
        )
        return self.best_box

    def update(self) -> Optional[BestBox]:
        self.update_distribution()
        return self.update_best_box()

    def get_best_box(self) -> Optional[BestBox]:
        return self.best_box


class IntentionTransformNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.robot = Robot(self, self.tf_buffer)
        plane = Plane(PLANE_NORMAL, PLANE_D)

        self.real_intention = Intention(
            node=self,
            name="real",
            robot=self.robot,
            box_topic=REAL_BOX_TOPIC,
            box_frame=BASE_FRAME,
            eef_target_frame=BASE_FRAME,
            plane=plane,
        )
        self.virtual_intention = Intention(
            node=self,
            name="virtual",
            robot=self.robot,
            box_topic=VIRTUAL_BOX_TOPIC,
            box_frame=MAP_FRAME,
            eef_target_frame=MAP_FRAME,
            plane=plane,
        )

        self.best_box_pub = self.create_publisher(
            MarkerArray,
            BEST_BOX_SELECTED_TOPIC,
            10,
        )

        self.joy_sub = self.create_subscription(
            Joy,
            JOY_TOPIC,
            self.joy_callback,
            10,
        )

        self.last_virtual_base_tf: Optional[TransformStamped] = None
        self.alignment_has_been_computed = False

        self.intention_timer = self.create_timer(
            1.0 / INTENTION_RATE_HZ,
            self.intention_timer_callback,
        )
        self.tf_timer = self.create_timer(
            1.0 / TF_RATE_HZ,
            self.tf_timer_callback,
        )

        self.get_logger().info(
            f"{NODE_NAME} started. real_topic={REAL_BOX_TOPIC}, "
            f"virtual_topic={VIRTUAL_BOX_TOPIC}, tool_frame={TOOL_FRAME}, "
            f"ignore_orientation={IGNORE_ORIENTATION}"
        )

    def joy_callback(self, msg: Joy) -> None:
        if RESET_BUTTON_INDEX >= len(msg.buttons):
            self.get_logger().warn(
                f"Joy message has {len(msg.buttons)} buttons, "
                f"but RESET_BUTTON_INDEX={RESET_BUTTON_INDEX}"
            )
            return

        if msg.buttons[RESET_BUTTON_INDEX] == 1:
            self.real_intention.reset()
            self.virtual_intention.reset()

    def intention_timer_callback(self) -> None:
        self.real_intention.update()
        self.virtual_intention.update()
        self.publish_selected_best_boxes()

    def publish_selected_best_boxes(self) -> None:
        real_best = self.real_intention.get_best_box()
        virtual_best = self.virtual_intention.get_best_box()

        marker_array = MarkerArray()
        stamp = now_msg(self)

        if real_best is not None:
            marker_array.markers.append(
                copy_marker_for_best_box(
                    real_best.box.marker,
                    BEST_BOX_REAL_MARKER_ID,
                    BEST_BOX_REAL_NS,
                    stamp,
                    ColorRGBA(r=1.0, g=0.4, b=0.0, a=0.8),
                )
            )

        if virtual_best is not None:
            marker_array.markers.append(
                copy_marker_for_best_box(
                    virtual_best.box.marker,
                    BEST_BOX_VIRTUAL_MARKER_ID,
                    BEST_BOX_VIRTUAL_NS,
                    stamp,
                    ColorRGBA(r=0.0, g=0.7, b=1.0, a=0.8),
                )
            )

        if marker_array.markers:
            self.best_box_pub.publish(marker_array)

    def compute_virtual_base_transform_from_best_boxes(
        self,
    ) -> Optional[TransformStamped]:
        real_best = self.real_intention.get_best_box()
        virtual_best = self.virtual_intention.get_best_box()

        if real_best is None or virtual_best is None:
            return None

        real_pose = real_best.box.pose
        virtual_pose = virtual_best.box.pose

        if IGNORE_ORIENTATION:
            mat = np.eye(4, dtype=float)
            mat[:3, 3] = np.array(
                [
                    virtual_pose.position.x - real_pose.position.x,
                    virtual_pose.position.y - real_pose.position.y,
                    virtual_pose.position.z - real_pose.position.z,
                ],
                dtype=float,
            )
        else:
            T_map_virtual_box = pose_to_matrix(virtual_pose, ignore_orientation=False)
            T_base_real_box = pose_to_matrix(real_pose, ignore_orientation=False)
            mat = T_map_virtual_box @ np.linalg.inv(T_base_real_box)

        return matrix_to_transform_stamped(
            mat,
            parent_frame=MAP_FRAME,
            child_frame=VIRTUAL_BASE_FRAME,
            stamp=now_msg(self),
        )

    def compute_fallback_transform_from_map_to_base(self) -> Optional[TransformStamped]:
        try:
            map_to_base = self.tf_buffer.lookup_transform(
                MAP_FRAME,
                BASE_FRAME,
                Time(),
                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"Fallback TF lookup failed {MAP_FRAME} -> {BASE_FRAME}: {exc}"
            )
            return None

        mat = transform_to_matrix(map_to_base)
        return matrix_to_transform_stamped(
            mat,
            parent_frame=MAP_FRAME,
            child_frame=VIRTUAL_BASE_FRAME,
            stamp=now_msg(self),
        )

    def publish_transform(self, transform: TransformStamped) -> None:
        # Always update stamp before broadcasting repeated/last transform.
        transform.header.stamp = now_msg(self)
        self.tf_broadcaster.sendTransform(transform)

    def tf_timer_callback(self) -> None:
        transform = self.compute_virtual_base_transform_from_best_boxes()

        if transform is not None:
            self.last_virtual_base_tf = transform
            self.alignment_has_been_computed = True
            self.publish_transform(transform)
            return

        if self.alignment_has_been_computed and self.last_virtual_base_tf is not None:
            self.publish_transform(self.last_virtual_base_tf)
            return

        fallback = self.compute_fallback_transform_from_map_to_base()
        if fallback is not None:
            self.last_virtual_base_tf = fallback
            self.publish_transform(fallback)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IntentionTransformNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
