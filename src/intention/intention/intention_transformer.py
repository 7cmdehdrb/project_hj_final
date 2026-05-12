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
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
    Transform,
    TransformStamped,
    Vector3,
)
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import ColorRGBA, Header
from rclpy.qos import QoSProfile, qos_profile_system_default
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# Replace this import path with the actual package/module path in your ROS2 workspace.
# The uploaded helper library provides these functions.
from rotutils import (
    compose_transform,
    decompose_transform,
    euler_from_quaternion,
    invert_transform,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_euler,
)

# =============================================================================
# Hard-coded configuration section
# =============================================================================
# Do not use ROS2 declare_parameter. Edit these variables directly.

NODE_NAME = "intention_transform_node"

REAL_BOX_TOPIC = "/real_box"
VIRTUAL_BOX_TOPIC = "/virtual_box"
BEST_BOX_SELECTED_TOPIC = "/best_box/selected"
DEBUG_MARKER_TOPIC = "/intention/debug_markers"

JOINT_STATE_TOPIC = "/joint_states"
JOY_TOPIC = "/controller/right/joy"
RESET_BUTTON_INDEX = 2

MAP_FRAME = "map"
BASE_FRAME = "base_link"
TOOL_FRAME = "tool0"
VIRTUAL_BASE_FRAME = "virtual_base_link"

# If True, box orientations are ignored during final TF alignment.
# Box poses are normalized to MAP_FRAME when they are received.
# Final alignment treats the selected real and virtual boxes as the same box in
# each robot's base frame:
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
MIN_TOOL_SPEED = 1e-4
MIN_GAIN_TOTAL = 1e-8
MAX_DEBUG_INTERSECTION_POINTS = 200
DEBUG_LOG_PERIOD_SEC = 1.0
PLANE_MARKER_SIZE = 1.2

# Plane generation mode.
# False: use the predefined plane below.
# True : estimate the plane from RealBox MarkerArray centers by least-squares/PCA.
USE_REGRESSED_PLANE_FROM_REAL_BOXES = False

# Predefined plane parameters from the previous implementation.
# Used either as the fixed plane, or as the initial/fallback plane before enough RealBox data arrives.
PLANE_NORMAL = np.array([1.0, 0.0, 0.0], dtype=float)
PLANE_D = -0.6

# At least 3 non-collinear points are required to fit a plane.
MIN_PLANE_REGRESSION_BOXES = 3

# TF lookup timeout.
TF_LOOKUP_TIMEOUT_SEC = 0.05

# Subscription QoS.
SUBSCRIPTION_QOS: QoSProfile = qos_profile_system_default

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


def pose_to_matrix(pose: Pose, ignore_orientation: bool = False) -> np.ndarray:
    translation = np.array(
        [pose.position.x, pose.position.y, pose.position.z],
        dtype=float,
    )

    if ignore_orientation:
        rotation = np.eye(3, dtype=float)
    else:
        q = pose.orientation
        roll, pitch, yaw = euler_from_quaternion(
            q.x,
            q.y,
            q.z,
            q.w,
        )
        rotation = rotation_matrix_from_euler(roll, pitch, yaw)

    return compose_transform(translation, rotation)


def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

    return compose_transform(
        translation=np.array([t.x, t.y, t.z], dtype=float),
        rotation=rotation_matrix_from_euler(roll, pitch, yaw),
    )


def matrix_to_transform_stamped(
    mat: np.ndarray,
    parent_frame: str,
    child_frame: str,
    stamp: TimeMsg,
) -> TransformStamped:
    translation, rotation = decompose_transform(np.asarray(mat, dtype=float))
    qx, qy, qz, qw = quaternion_from_rotation_matrix(rotation)

    return TransformStamped(
        header=Header(stamp=stamp, frame_id=parent_frame),
        child_frame_id=child_frame,
        transform=Transform(
            translation=Vector3(
                x=float(translation[0]),
                y=float(translation[1]),
                z=float(translation[2]),
            ),
            rotation=Quaternion(
                x=float(qx),
                y=float(qy),
                z=float(qz),
                w=float(qw),
            ),
        ),
    )


def matrix_to_pose(mat: np.ndarray) -> Pose:
    translation, rotation = decompose_transform(np.asarray(mat, dtype=float))
    qx, qy, qz, qw = quaternion_from_rotation_matrix(rotation)

    return Pose(
        position=Point(
            x=float(translation[0]),
            y=float(translation[1]),
            z=float(translation[2]),
        ),
        orientation=Quaternion(
            x=float(qx),
            y=float(qy),
            z=float(qz),
            w=float(qw),
        ),
    )


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


def make_delete_all_marker(stamp: TimeMsg, frame_id: str) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=stamp, frame_id=frame_id)
    marker.action = Marker.DELETEALL
    return marker


def make_plane_marker(
    plane: "Plane",
    frame_id: str,
    stamp: TimeMsg,
    marker_id: int,
    namespace: str,
    color: ColorRGBA,
    size: float = PLANE_MARKER_SIZE,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=stamp, frame_id=frame_id)
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale = Vector3(x=1.0, y=1.0, z=1.0)
    marker.color = color

    center = -plane.d * plane.n
    u = np.array([-plane.n[1], plane.n[0], 0.0], dtype=float)
    if np.linalg.norm(u) < 1e-12:
        u = np.array([0.0, -plane.n[2], plane.n[1]], dtype=float)
    u = u / np.linalg.norm(u)
    v = np.cross(plane.n, u)
    v = v / np.linalg.norm(v)

    half = float(size) * 0.5
    corners = [
        center - half * u - half * v,
        center + half * u - half * v,
        center + half * u + half * v,
        center - half * u + half * v,
    ]
    triangles = [corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]]
    marker.points = [
        Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in triangles
    ]
    return marker


def make_points_marker(
    points: np.ndarray,
    frame_id: str,
    stamp: TimeMsg,
    marker_id: int,
    namespace: str,
    color: ColorRGBA,
    point_scale: float = 0.025,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=stamp, frame_id=frame_id)
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale = Vector3(x=point_scale, y=point_scale, z=point_scale)
    marker.color = color

    if points.size > 0:
        valid = points[~np.isnan(points).any(axis=1)]
        valid = valid[-MAX_DEBUG_INTERSECTION_POINTS:]
        marker.points = [
            Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in valid
        ]
    return marker


def make_sphere_marker(
    point: np.ndarray,
    frame_id: str,
    stamp: TimeMsg,
    marker_id: int,
    namespace: str,
    color: ColorRGBA,
    scale: float = 0.055,
) -> Marker:
    marker = Marker()
    marker.header = Header(stamp=stamp, frame_id=frame_id)
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position = Point(
        x=float(point[0]), y=float(point[1]), z=float(point[2])
    )
    marker.pose.orientation.w = 1.0
    marker.scale = Vector3(x=scale, y=scale, z=scale)
    marker.color = color
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
        self.set(n, d)

    def set(self, n: np.ndarray, d: float) -> None:
        if not isinstance(n, np.ndarray):
            raise ValueError("n must be a numpy array")
        if n.shape != (3,):
            raise ValueError("n must have shape (3,)")

        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError("plane normal vector must be non-zero")

        self.n = n.astype(float) / norm
        self.d = float(d) / norm

    def fit_from_boxes(self, boxes: Dict[int, BoxRecord]) -> bool:
        if len(boxes) < MIN_PLANE_REGRESSION_BOXES:
            return False

        points = np.array(
            [
                [box.pose.position.x, box.pose.position.y, box.pose.position.z]
                for box in boxes.values()
            ],
            dtype=float,
        )

        if points.shape[0] < MIN_PLANE_REGRESSION_BOXES:
            return False

        centroid = np.mean(points, axis=0)
        centered = points - centroid

        try:
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return False

        if vh.shape[0] < 3:
            return False

        normal = vh[-1, :]
        if np.linalg.norm(normal) < 1e-12:
            return False

        # Use box orientations only to choose the normal direction consistently.
        # The local +X axis of each box pose is treated as the box-plane normal candidate.
        orientation_normals = []
        for box in boxes.values():
            q = box.pose.orientation
            roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            rotation = rotation_matrix_from_euler(roll, pitch, yaw)
            orientation_normals.append(rotation[:, 0])

        if orientation_normals:
            mean_orientation_normal = np.mean(np.array(orientation_normals), axis=0)
            if np.linalg.norm(mean_orientation_normal) > 1e-12:
                if np.dot(normal, mean_orientation_normal) < 0.0:
                    normal = -normal

        d = -float(np.dot(normal, centroid))
        self.set(normal, d)
        return True

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
    def __init__(
        self,
        node: Node,
        topic: str,
        default_source_frame: str,
        target_frame: str,
        tf_buffer: Buffer,
        name: str,
    ):
        self.node = node
        self.topic = topic
        self.default_source_frame = default_source_frame
        self.target_frame = target_frame
        self.tf_buffer = tf_buffer
        self.name = name
        self.boxes: Dict[int, BoxRecord] = {}

        self.sub = node.create_subscription(
            MarkerArray,
            topic,
            self.callback,
            qos_profile=SUBSCRIPTION_QOS,
        )

    def transform_marker_to_target_frame(self, marker: Marker) -> Optional[Marker]:
        source_frame = marker.header.frame_id or self.default_source_frame
        marker_in_target = copy.deepcopy(marker)

        if source_frame != self.default_source_frame:
            self.node.get_logger().warn(
                f"[{self.name}] marker frame_id='{source_frame}' differs from "
                f"default_source_frame='{self.default_source_frame}'. "
                f"The marker will still be transformed to '{self.target_frame}'."
            )

        if source_frame == self.target_frame:
            marker_in_target.header.frame_id = self.target_frame
            return marker_in_target

        try:
            target_to_source = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC),
            )
        except Exception as exc:
            self.node.get_logger().warn(
                f"[{self.name}] failed to transform marker id={marker.id} "
                f"from '{source_frame}' to '{self.target_frame}': {exc}"
            )
            return None

        T_target_source = transform_to_matrix(target_to_source)
        T_source_marker = pose_to_matrix(marker.pose, ignore_orientation=False)
        marker_in_target.pose = matrix_to_pose(T_target_source @ T_source_marker)
        marker_in_target.header.frame_id = self.target_frame
        return marker_in_target

    def callback(self, msg: MarkerArray) -> None:
        # The incoming MarkerArray is a full fresh observation, but missing markers
        # are intentionally not deleted because transient perception dropout is possible.
        for marker in msg.markers:
            marker: Marker

            if marker.action == Marker.DELETEALL:
                continue
            if marker.action == Marker.DELETE:
                continue

            marker_in_target = self.transform_marker_to_target_frame(marker)
            if marker_in_target is None:
                continue

            self.boxes[int(marker.id)] = BoxRecord(
                id=int(marker.id),
                pose=copy.deepcopy(marker_in_target.pose),
                frame_id=self.target_frame,
                marker=marker_in_target,
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
            qos_profile=SUBSCRIPTION_QOS,
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
        except Exception as exc:
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
            trust_threshold = max(0.0, float(trust_threshold))
            distrust_threshold = max(0.0, float(distrust_threshold))
            if distrust_threshold < trust_threshold:
                trust_threshold, distrust_threshold = (
                    distrust_threshold,
                    trust_threshold,
                )

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
            v_norm = float(np.linalg.norm(v))
            if v_norm < MIN_TOOL_SPEED:
                return 0.0

            # Use the acute angle to the plane normal. The previous signed-normal
            # angle could approach pi when the tool moved toward the opposite side
            # of the same plane, which drove gain_theta to 0 even for valid motion.
            denominator = v_norm * float(np.linalg.norm(plane.n))
            cos_value = float(np.clip(abs(np.dot(v, plane.n)) / denominator, -1.0, 1.0))
            return float(np.arccos(cos_value))

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
                theta_d_min = max(0.0, float(theta_d_min))
                theta_d_max = max(theta_d_min, float(theta_d_max))
                return theta_d_min, theta_d_max

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
        self.box_manager = BoxManager(
            node=node,
            topic=box_topic,
            default_source_frame=box_frame,
            target_frame=MAP_FRAME,
            tf_buffer=robot.tf_buffer,
            name=name,
        )
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
        self._last_log_times: Dict[str, float] = {}

    def _log_throttled(
        self,
        key: str,
        level: str,
        message: str,
        period_sec: float = DEBUG_LOG_PERIOD_SEC,
    ) -> None:
        """Log with simple time throttling.

        Do not call logger methods through getattr() from a single source line.
        rclpy associates throttled logger call sites with severity, and using the
        same call site for both INFO and WARN can raise:
        ValueError: Logger severity cannot be changed between calls.
        """
        return
        now_sec = self.node.get_clock().now().nanoseconds * 1e-9
        last_sec = self._last_log_times.get(key, -1.0e30)
        if now_sec - last_sec < period_sec:
            return
        self._last_log_times[key] = now_sec

        logger = self.node.get_logger()
        if level == "debug":
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level == "warn" or level == "warning":
            logger.warn(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

    def get_debug_points(self) -> np.ndarray:
        return self.intersections.copy()

    def get_mean_3d_for_debug(self) -> Optional[np.ndarray]:
        if (
            not self.has_distribution
            or len(self.intersections) < MIN_INTERSECTION_SAMPLES
        ):
            return None
        weight_sum = float(np.sum(self.gains))
        if weight_sum <= MIN_GAIN_TOTAL:
            return None
        try:
            return np.average(self.intersections, axis=0, weights=self.gains)
        except Exception:
            return None

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
            self._log_throttled(
                "no_tool_state",
                "warn",
                f"[{self.name}] cannot update distribution: TF for {self.eef_target_frame} -> {TOOL_FRAME} is unavailable",
            )
            return False

        p = tool_state.position
        v = tool_state.linear_velocity
        speed = float(np.linalg.norm(v))
        if speed < MIN_TOOL_SPEED:
            self._log_throttled(
                "low_speed",
                "info",
                f"[{self.name}] distribution not updated: tool speed is too small "
                f"speed={speed:.6e}, p={p.tolist()}",
            )
            return False

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
        gain_theta = (
            self.IntersectionMethod.get_gain_theta(
                theta_between_plane,
                trust_theta,
                distrust_theta,
            )
            + 1e-6
        )  # Avoid zero gain when theta is slightly above distrust threshold.
        gain_total = gain_distance * gain_theta
        print(
            f"Gain total: {gain_total:.3e}, Gain distance: {gain_distance:.3e}, Gain theta: {gain_theta:.3e}"
        )

        if np.isnan(intersection).any():
            print(
                f"Invalid intersection: v is nearly parallel to plane or contains NaN. "
                f"speed={speed:.6e}, distance={distance_between_plane:.6f}, "
                f"theta={theta_between_plane:.6f}, plane_n={self.plane.n.tolist()}, d={self.plane.d:.6f}"
            )

            self._log_throttled(
                "invalid_intersection",
                "warn",
                f"[{self.name}] invalid intersection: v is nearly parallel to plane or contains NaN. "
                f"speed={speed:.6e}, distance={distance_between_plane:.6f}, "
                f"theta={theta_between_plane:.6f}, plane_n={self.plane.n.tolist()}, d={self.plane.d:.6f}",
            )
            return False

        if gain_total <= MIN_GAIN_TOTAL:
            self._log_throttled(
                "zero_gain",
                "warn",
                f"[{self.name}] intersection rejected because gain is too small. "
                f"gain_total={gain_total:.3e}, gain_distance={gain_distance:.3e}, "
                f"gain_theta={gain_theta:.3e}, theta={theta_between_plane:.4f}, "
                f"trust={trust_theta:.4f}, distrust={distrust_theta:.4f}, "
                f"forward={forward}, distance={distance_between_plane:.4f}, "
                f"p={p.tolist()}, v={v.tolist()}, intersection={intersection.tolist()}",
            )
            print(
                f"Rejected intersection: gain_total={gain_total:.3e}, gain_distance={gain_distance:.3e}, "
            )
            return False

        if forward:
            print(
                "Forward intersection accepted: gain_total={gain_total:.3e}, gain_distance={gain_distance:.3e}, "
            )
            if self.last_direction_forward != forward:
                if len(self.reverse_distances) > 2:
                    max_reverse_dist = np.max(self.reverse_distances)
                    mask = self.distances > max_reverse_dist
                    removed = int(len(self.intersections) - np.count_nonzero(mask))
                    self.intersections = self.intersections[mask]
                    self.gains = self.gains[mask]
                    self.distances = self.distances[mask]
                    self._log_throttled(
                        "reverse_prune",
                        "info",
                        f"[{self.name}] pruned stale intersections after direction change: "
                        f"removed={removed}, remaining={len(self.intersections)}",
                    )

                self.reverse_distances = np.empty(0, dtype=float)

            self.intersections = np.vstack([self.intersections, intersection])
            self.gains = np.append(self.gains, gain_total)
            self.distances = np.append(self.distances, distance_between_plane)
            self._log_throttled(
                "accepted_intersection",
                "info",
                f"[{self.name}] accepted intersection: samples={len(self.intersections)}, "
                f"weight_sum={float(np.sum(self.gains)):.3e}, gain={gain_total:.3e}, "
                f"distance={distance_between_plane:.4f}, theta={theta_between_plane:.4f}",
            )
        else:
            self.reverse_distances = np.append(
                self.reverse_distances, distance_between_plane
            )
            self._log_throttled(
                "behind_plane",
                "info",
                f"[{self.name}] intersection is behind current tool motion; not appended. "
                f"reverse_samples={len(self.reverse_distances)}, distance={distance_between_plane:.4f}",
            )

        self.last_direction_forward = forward

        if len(self.intersections) < MIN_INTERSECTION_SAMPLES:
            self._log_throttled(
                "not_enough_samples",
                "info",
                f"[{self.name}] waiting for more valid intersections: "
                f"samples={len(self.intersections)}/{MIN_INTERSECTION_SAMPLES}",
            )
            return False

        weight_sum = float(np.sum(self.gains))
        if weight_sum <= MIN_GAIN_TOTAL:
            self._log_throttled(
                "weight_sum_small",
                "warn",
                f"[{self.name}] distribution update skipped: weight_sum is too small. "
                f"samples={len(self.intersections)}, weight_sum={weight_sum:.3e}, "
                f"min_gain={float(np.min(self.gains)):.3e}, max_gain={float(np.max(self.gains)):.3e}",
            )
            return False

        try:
            mean_3d, cov_3d = self.IntersectionMethod.get_weight_mean_and_covariance(
                self.intersections,
                self.gains,
            )
            mean_2d, cov_2d = self.plane.transform_to_2d(mean_3d, cov_3d)

            print(f"Mean 2D: {self.mean_2d}, Cov 2D: {self.cov_2d}")
        except Exception as exc:
            self.node.get_logger().warn(
                f"[{self.name}] distribution update failed: {exc}. "
                f"samples={len(self.intersections)}, weight_sum={weight_sum:.3e}, "
                f"speed={speed:.6e}, distance={distance_between_plane:.6f}, "
                f"theta={theta_between_plane:.6f}, trust={trust_theta:.6f}, "
                f"distrust={distrust_theta:.6f}"
            )
            return False

        self.mean_2d = mean_2d
        self.cov_2d = cov_2d + np.eye(2) * 1e-9
        self.has_distribution = True
        self._log_throttled(
            "distribution_ok",
            "info",
            f"[{self.name}] distribution updated: mean_2d={self.mean_2d.tolist()}, "
            f"cov_diag={np.diag(self.cov_2d).tolist()}, samples={len(self.intersections)}, "
            f"weight_sum={weight_sum:.3e}",
        )

        print(f"Mean 2D: {self.mean_2d}, Cov 2D: {self.cov_2d}")

        return True

    def update_best_box(self) -> Optional[BestBox]:
        if not self.has_distribution:
            self._log_throttled(
                "no_distribution_for_best_box",
                "info",
                f"[{self.name}] best box not updated: distribution is not ready",
            )
            return self.best_box

        boxes = self.box_manager.get_boxes()
        if not boxes:
            self._log_throttled(
                "no_boxes",
                "warn",
                f"[{self.name}] best box not updated: no boxes received on {self.box_manager.topic}",
            )
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

        self.plane = plane

        self.real_intention = Intention(
            node=self,
            name="real",
            robot=self.robot,
            box_topic=REAL_BOX_TOPIC,
            box_frame=BASE_FRAME,
            eef_target_frame=MAP_FRAME,
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
        self.debug_marker_pub = self.create_publisher(
            MarkerArray,
            DEBUG_MARKER_TOPIC,
            10,
        )
        self._last_node_log_times: Dict[str, float] = {}

        self.joy_sub = self.create_subscription(
            Joy,
            JOY_TOPIC,
            self.joy_callback,
            qos_profile=SUBSCRIPTION_QOS,
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

        self._is_plane_updated = False
        self.get_logger().info(
            f"{NODE_NAME} started. real_topic={REAL_BOX_TOPIC}, "
            f"virtual_topic={VIRTUAL_BOX_TOPIC}, tool_frame={TOOL_FRAME}, "
            f"ignore_orientation={IGNORE_ORIENTATION}, debug_marker_topic={DEBUG_MARKER_TOPIC}"
        )

    def _log_throttled(
        self,
        key: str,
        level: str,
        message: str,
        period_sec: float = DEBUG_LOG_PERIOD_SEC,
    ) -> None:
        """Log with simple time throttling using severity-specific call sites."""
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        last_sec = self._last_node_log_times.get(key, -1.0e30)
        if now_sec - last_sec < period_sec:
            return
        self._last_node_log_times[key] = now_sec

        logger = self.get_logger()
        if level == "debug":
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level == "warn" or level == "warning":
            logger.warn(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

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
        if self._is_plane_updated is False:
            self.update_plane_if_enabled()

        self.real_intention.update()
        self.virtual_intention.update()
        self.publish_selected_best_boxes()
        self.publish_debug_markers()

    def update_plane_if_enabled(self) -> None:
        if not USE_REGRESSED_PLANE_FROM_REAL_BOXES:
            return

        real_boxes = self.real_intention.box_manager.get_boxes()
        if len(real_boxes) < MIN_PLANE_REGRESSION_BOXES:
            self._log_throttled(
                "plane_waiting_boxes",
                "info",
                f"Plane regression waiting for boxes: "
                f"real_boxes={len(real_boxes)}/{MIN_PLANE_REGRESSION_BOXES}. "
                f"Using fallback/current plane n={self.plane.n.tolist()}, d={self.plane.d:.6f}",
            )
            return

        updated = self.plane.fit_from_boxes(real_boxes)
        if updated:
            self._log_throttled(
                "plane_updated",
                "info",
                f"Regressed plane updated: n={self.plane.n.tolist()}, d={self.plane.d:.6f}, "
                f"real_boxes={len(real_boxes)}",
            )
            self._is_plane_updated = True
        else:
            self._log_throttled(
                "plane_failed",
                "warn",
                f"Plane regression failed. Check whether real box centers are nearly collinear. "
                f"real_boxes={len(real_boxes)}",
            )

    def publish_debug_markers(self) -> None:
        stamp = now_msg(self)
        marker_array = MarkerArray()
        marker_array.markers.append(make_delete_all_marker(stamp, MAP_FRAME))

        marker_array.markers.append(
            make_plane_marker(
                self.plane,
                MAP_FRAME,
                stamp,
                10,
                "estimated_plane_map_frame_real",
                ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.22),
            )
        )
        marker_array.markers.append(
            make_plane_marker(
                self.plane,
                MAP_FRAME,
                stamp,
                11,
                "estimated_plane_map_frame",
                ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.18),
            )
        )

        marker_array.markers.append(
            make_points_marker(
                self.real_intention.get_debug_points(),
                MAP_FRAME,
                stamp,
                20,
                "real_intersection_points",
                ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.95),
            )
        )
        marker_array.markers.append(
            make_points_marker(
                self.virtual_intention.get_debug_points(),
                MAP_FRAME,
                stamp,
                21,
                "virtual_intersection_points",
                ColorRGBA(r=0.1, g=0.4, b=1.0, a=0.95),
            )
        )

        real_mean = self.real_intention.get_mean_3d_for_debug()
        if real_mean is not None:
            marker_array.markers.append(
                make_sphere_marker(
                    real_mean,
                    MAP_FRAME,
                    stamp,
                    30,
                    "real_intersection_weighted_mean",
                    ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                )
            )

        virtual_mean = self.virtual_intention.get_mean_3d_for_debug()
        if virtual_mean is not None:
            marker_array.markers.append(
                make_sphere_marker(
                    virtual_mean,
                    MAP_FRAME,
                    stamp,
                    31,
                    "virtual_intersection_weighted_mean",
                    ColorRGBA(r=0.0, g=0.2, b=1.0, a=1.0),
                )
            )

        self.debug_marker_pub.publish(marker_array)

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

        if real_best is None and virtual_best is None:
            return self.compute_fallback_transform_from_map_to_base()

        if real_best is None or virtual_best is None:
            return None

        try:
            base_from_map = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                MAP_FRAME,
                Time(),
                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_SEC),
            )
        except Exception as exc:
            self._log_throttled(
                "virtual_base_alignment_base_map_lookup_failed",
                "warn",
                f"Failed to lookup TF {BASE_FRAME} -> {MAP_FRAME} "
                f"for virtual base alignment: {exc}",
            )
            return None

        T_map_real_box = pose_to_matrix(
            real_best.box.pose,
            ignore_orientation=IGNORE_ORIENTATION,
        )
        T_map_virtual_box = pose_to_matrix(
            virtual_best.box.pose,
            ignore_orientation=IGNORE_ORIENTATION,
        )
        T_base_real_box = transform_to_matrix(base_from_map) @ T_map_real_box

        if IGNORE_ORIENTATION:
            mat = np.eye(4, dtype=float)
            mat[:3, 3] = T_map_virtual_box[:3, 3] - T_base_real_box[:3, 3]
        else:
            mat = T_map_virtual_box @ invert_transform(T_base_real_box)

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
        except Exception as exc:
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
