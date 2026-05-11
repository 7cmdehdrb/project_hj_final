#!/usr/bin/python3

# Basic modules
import os
import sys
import rospy
import roslib
import numpy as np

# TF modules
import tf
from tf.transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_inverse,
    quaternion_multiply,
)

# Messages
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import *

# Custom
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.integrate import quad
from enum import Enum

try:
    sys.path.append(roslib.packages.get_pkg_dir("test_package") + "/src")
    from abstract_state import State
except roslib.exceptions.ROSLibException:
    rospy.logfatal("Cannot find package test_package")
    sys.exit(1)


class BestBox:
    def __init__(self, box: BoxObject, possibility: float):
        self.box = box
        self.possibility = possibility


class Plane(object):
    def __init__(self, n: np.array, d: float):
        if not isinstance(n, np.ndarray):
            raise ValueError("n must be a numpy array")

        self.n = n
        self.d = d

    def distance(self, p: np.array):
        """Calculate the distance of a point to the plane"""
        return np.abs(np.dot(self.n, p) + self.d) / np.linalg.norm(self.n)

    def project_to_plane(self, p: np.array):
        """Project a 3D point onto the plane and return its 2D coordinates"""
        # Find a point on the plane
        point_on_plane = p - (np.dot(self.n, p) + self.d) * self.n

        # Create a local coordinate system on the plane
        u = np.array([-self.n[1], self.n[0], 0])  # Arbitrary vector orthogonal to n
        if np.allclose(u, 0):
            u = np.array([0, -self.n[2], self.n[1]])  # Handle edge case
        u = u / np.linalg.norm(u)  # Normalize
        v = np.cross(self.n, u)  # Ensure v is orthogonal to both n and u

        # Project the point onto the local 2D basis
        x = np.dot(point_on_plane, u)
        y = np.dot(point_on_plane, v)
        return np.array([x, y]), u, v

    def transform_covariance(self, cov_3d: np.array, u: np.array, v: np.array):
        """Transform a 3D covariance matrix to 2D on the plane"""
        # Transformation matrix from 3D to 2D
        T = np.stack((u, v), axis=0)  # Shape: (2, 3)
        # Project the covariance matrix
        cov_2d = T @ cov_3d @ T.T  # Shape: (2, 2)
        return cov_2d

    def transform_to_2d(self, position: np.array, covariance: np.array):
        """Transform a 3D position and covariance to 2D on the plane"""
        if not isinstance(position, np.ndarray) or not isinstance(
            covariance, np.ndarray
        ):
            raise ValueError("Position and covariance must be numpy arrays")
        if position.shape != (3,) or covariance.shape != (3, 3):
            raise ValueError(
                "Position must be a 3-element vector and covariance must be 3x3"
            )

        # Project position to 2D
        pos_2d, u, v = self.project_to_plane(position)

        # Transform covariance to 2D
        cov_2d = self.transform_covariance(covariance, u, v)

        return pos_2d, cov_2d


class BoxManager:
    def __init__(self, topic: str):
        self.__sub = rospy.Subscriber(topic, BoxObjectMultiArray, self.callback)
        self.__box_pub = rospy.Publisher(
            "boxes/virtual/visual", MarkerArray, queue_size=10
        )

        self.data = []  # [BoxObject]

    def callback(self, msg: BoxObjectMultiArray):
        new_data = []

        for new_box in msg.boxes:
            new_data.append(new_box)

        self.data = new_data

    def publish_box(self):
        marker_array = MarkerArray()

        for box in self.data:
            box: BoxObject

            marker = Marker()
            marker.header = Header(stamp=rospy.Time.now(), frame_id="base_link")
            marker.id = box.id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = box.pose.position
            marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            marker.scale = Vector3(x=0.01, y=0.25, z=0.15)
            marker.color = ColorRGBA(r=255 / 255, g=195 / 255, b=0, a=0.3)

            marker_array.markers.append(marker)

        self.__box_pub.publish(marker_array)


class EEF_Twist_Subscriber:
    def __init__(self, topic: str):
        self.__sub = rospy.Subscriber(topic, Twist, self.callback)
        self.data = Twist()

    def callback(self, msg: Twist):
        self.data = msg


class RealIntentionGaussian:
    class IntersectionMethod:
        @staticmethod
        def get_gain_distance(d, a=1.0):
            return 1 + (1 / (a * d))

        def get_gain_theta(theta, trust_threshold, distrust_threshold):
            theta = np.abs(theta)
            if theta < trust_threshold:
                return 1.0
            elif trust_threshold <= theta and theta < distrust_threshold:
                return np.cos(
                    (np.pi / (2.0 * (distrust_threshold - trust_threshold)))
                    * (theta - trust_threshold)
                )
            else:
                return 0.0

        @staticmethod
        def get_theta(v: np.array, plane: Plane):
            if np.isclose(np.linalg.norm(v), 0):
                return 0.0

            numerator = np.dot(v, plane.n)
            demoninator = np.linalg.norm(v) * np.linalg.norm(plane.n)

            angle = np.arccos(numerator / demoninator)

            angle = angle % (2 * np.pi)

            return angle

        @staticmethod
        def get_intersection_point(p: np.array, v: np.array, plane: Plane):
            denominator = np.dot(plane.n, v)
            numerator = plane.n @ p + plane.d
            if np.isclose(denominator, 0):  # 분모가 0에 가까우면 평행한 상태
                return True, np.array([np.nan, np.nan, np.nan])

            t = -numerator / denominator
            return (t > 0.0 and t < 30.0), p + (t * v)

        @staticmethod
        def get_theta(v: np.array, plane: Plane):
            if np.isclose(np.linalg.norm(v), 0):
                return 0.0

            numerator = np.dot(v, plane.n)
            demoninator = np.linalg.norm(v) * np.linalg.norm(plane.n)

            angle = np.arccos(numerator / demoninator)

            angle = angle % (2 * np.pi)

            return angle

        @staticmethod
        def get_weight_mean_and_covariance(data: np.array, weights: np.array):
            weighted_mean = np.average(data, axis=0, weights=weights)
            centered_data = data - weighted_mean

            weighted_covariance = np.cov(centered_data.T, aweights=weights, bias=True)

            return weighted_mean, weighted_covariance

        @staticmethod
        def get_trust_theta(p: np.array, plane: Plane):
            R = 1.125  # manipulator length
            D = plane.distance(
                np.array([0, 0, 0])
            )  # Box plane과 manipulator base_link 사이 거리
            d = plane.distance(p)  # Box plane과 manipulator eef 사이 거리

            cls = RealIntentionGaussian.IntersectionMethod

            check_eef_intersection, eef_intersection_points = (
                cls.get_intersection_point(p, plane.n, plane)
            )
            check_base_intersection, base_intersection_points = (
                cls.get_intersection_point(np.array([0, 0, 0]), plane.n, plane)
            )

            if check_eef_intersection and check_base_intersection:
                theta_D = np.arccos(D / R)

                theta_d_max = np.arctan(
                    (
                        np.abs(D * np.tan(theta_D))
                        + np.abs(
                            np.linalg.norm(
                                base_intersection_points[1:]
                                - eef_intersection_points[1:]
                            )
                        )
                    )
                    / d
                )
                theta_d_min = np.arctan(
                    (
                        np.abs(D * np.tan(theta_D))
                        - np.abs(
                            np.linalg.norm(
                                base_intersection_points[1:]
                                - eef_intersection_points[1:]
                            )
                        )
                    )
                    / d
                )
                return float(theta_d_min), float(theta_d_max)

            return 0.0, 0.0

    def __init__(self):
        # Define EEF and Boxes
        self.eef_pose_state = State(topic=None, frame_id="VGC_virtual")
        self.eef_twist_state = EEF_Twist_Subscriber("/calculated_twist")
        self.box_manager = BoxManager("/virtual_box")

        # Variables
        self.mean = np.array([0, 0])
        self.cov = np.eye(2)

        self.intersections = np.empty((0, 3))  # Projection data
        self.gains = np.empty(0)

        self.distances = np.empty(0)
        self.reverse_distances = np.empty(0)

        self.best_box = None

        self.plane = Plane(
            n=np.array([0.99887537, 0.0465492, 0.00900962]), d=-0.8969238154115642
        )

        # Parameters
        self.R = 1.125
        self.a = 1.0
        self.last_direction = True

        # ROS
        self.right_controller_sub = rospy.Subscriber(
            "/controller/right/joy", Joy, self.right_controller_callback
        )  # Trigger to reset
        self.best_box_pub = rospy.Publisher(
            "/best_box/virtual", BoxObject, queue_size=10
        )

        # For Final
        self.eef_to_plane = 0.0
        self.distance_pub = rospy.Publisher("/distance/virtual", Float32, queue_size=10)
        self.intersections_pub = rospy.Publisher(
            "/intersection/virtual", PoseWithCovarianceStamped, queue_size=10
        )
        self.pdf_pub = rospy.Publisher(
            "/pdf/virtual", BoxObjectWithPDFMultiArray, queue_size=10
        )

    def reset(self):
        """Reset data"""
        self.intersections = np.empty((0, 3))
        self.gains = np.empty(0)
        self.distances = np.empty(0)
        self.reverse_distances = np.empty(0)

        self.mean = np.array([0, 0])
        self.cov = np.eye(2)

        self.best_box = None

    def right_controller_callback(self, msg: Joy):
        if len(msg.buttons) != 4:
            rospy.logwarn("Invalid Joy Message.")
            return

        if msg.buttons[2] == 1:
            self.reset()

    def get_mean_and_covariance(self):
        p = self.eef_pose_state.get_target_frame_pose2(target_frame="map")
        v = self.eef_twist_state.data.linear

        if p is None:
            rospy.logwarn("No target frame pose")
            return None, None

        p: PoseStamped
        p = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
        v = np.array([v.x, v.y, v.z])

        distance_between_plane = self.plane.distance(p)
        theta_between_plane = RealIntentionGaussian.IntersectionMethod.get_theta(
            v, self.plane
        )

        forward, intersection = (
            RealIntentionGaussian.IntersectionMethod.get_intersection_point(
                p, v, self.plane
            )
        )
        trust_theta, distrust_theta = (
            RealIntentionGaussian.IntersectionMethod.get_trust_theta(p, self.plane)
        )
        gain_distance = RealIntentionGaussian.IntersectionMethod.get_gain_distance(
            distance_between_plane, self.a
        )
        gain_theta = RealIntentionGaussian.IntersectionMethod.get_gain_theta(
            theta_between_plane, trust_theta, distrust_theta
        )
        gain_total = gain_distance * gain_theta

        mean_2d = None
        cov_2d = None

        if np.isnan(intersection).any():
            # rospy.logwarn("No intersection")
            return None, None

        if forward:
            # Forward

            if self.last_direction != forward:
                # Change direction

                if len(self.reverse_distances) > 2:
                    max_reverse_dist = np.max(self.reverse_distances)
                    mask = self.distances > max_reverse_dist

                    self.intersections = self.intersections[mask]
                    self.gains = self.gains[mask]
                    self.distances = self.distances[mask]

                self.reverse_distances = np.empty(0)

            try:
                mean, cov = (
                    RealIntentionGaussian.IntersectionMethod.get_weight_mean_and_covariance(
                        self.intersections, self.gains
                    )
                )

                mean_2d, cov_2d = self.plane.transform_to_2d(mean, cov)

            except Exception as e:
                rospy.logwarn(e)

            self.intersections = np.vstack([self.intersections, intersection])
            self.gains = np.append(self.gains, gain_total)
            self.distances = np.append(self.distances, distance_between_plane)

        else:
            self.reverse_distances = np.append(
                self.reverse_distances, distance_between_plane
            )

        self.last_direction = forward

        return mean_2d, cov_2d

    def publish_result(self):
        mean_2d, cov_2d = self.get_mean_and_covariance()

        if mean_2d is not None and cov_2d is not None:
            self.mean = mean_2d
            self.cov = cov_2d

        if (
            self.cov[0, 0] == 1.0
            or self.cov[1, 1] == 1.0
            or self.mean[0] == 0.0
            or self.mean[1] == 0.0
        ):
            rospy.logwarn("No mean and covariance")
            return None, None

        # Make PoseWithCovarianceStamped
        pose_with_cov = PoseWithCovarianceStamped()

        pose_with_cov.header = Header(stamp=rospy.Time.now(), frame_id="map")
        pose_with_cov.pose.pose.position = Point(
            x=-(self.plane.d), y=self.mean[0], z=self.mean[1]
        )

        cov = np.zeros((6, 6))
        cov[1:3, 1:3] = self.cov
        cov = cov.flatten().tolist()

        pose_with_cov.pose.covariance = cov

        # Make PDF
        boxes_with_pdf = BoxObjectWithPDFMultiArray()

        rv = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True)
        t_max_pdf = rv.pdf(self.mean)

        max_pdf = 0.0
        best_box = None

        for box in self.box_manager.data:
            box: BoxObject
            box_with_pdf = BoxObjectWithPDF()

            box_p = np.array(
                [box.pose.position.x, box.pose.position.y, box.pose.position.z]
            )
            box_p_2d, _, _ = self.plane.project_to_plane(box_p)

            pdf = rv.pdf(box_p_2d)

            box_with_pdf.id = box.id
            box_with_pdf.pdf = pdf

            boxes_with_pdf.boxes.append(box_with_pdf)

        self.distance_pub.publish(Float32(data=self.eef_to_plane))
        self.intersections_pub.publish(pose_with_cov)
        self.pdf_pub.publish(boxes_with_pdf)

    def get_best_box(self):
        mean_2d, cov_2d = self.get_mean_and_covariance()

        if mean_2d is not None and cov_2d is not None:
            self.mean = mean_2d
            self.cov = cov_2d

        if (
            self.cov[0, 0] == 1.0
            or self.cov[1, 1] == 1.0
            or self.mean[0] == 0.0
            or self.mean[1] == 0.0
        ):
            rospy.logwarn("No mean and covariance")
            return None, None

        rv = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True)
        t_max_pdf = rv.pdf(self.mean)

        max_pdf = 0.0
        best_box = None

        for box in self.box_manager.data:
            box: BoxObject

            box_p = np.array(
                [box.pose.position.x, box.pose.position.y, box.pose.position.z]
            )
            box_p_2d, _, _ = self.plane.project_to_plane(box_p)

            pdf = rv.pdf(box_p_2d)

            if pdf > max_pdf:
                max_pdf = pdf
                best_box = box

        return (max_pdf / t_max_pdf), best_box

    def update(self):
        possibility, best_box = self.get_best_box()

        if best_box is None:
            rospy.logwarn("No best box")
            return None

        self.best_box = BestBox(best_box, possibility)

        if self.best_box is not None:
            rospy.loginfo(
                f"Num of Boxes: {len(self.box_manager.data)}, Best Box: {self.best_box.box.id}, Possibility: {self.best_box.possibility}"
            )
            self.best_box_pub.publish(self.best_box.box)


def main():
    rospy.init_node("virtual_intention_gaussian_node")  # TODO: Add node name

    intention = RealIntentionGaussian()

    r = rospy.Rate(10)  # TODO: Add rate
    while not rospy.is_shutdown():

        intention.publish_result()
        intention.update()

        r.sleep()


if __name__ == "__main__":
    pass
    try:
        main()
    except rospy.ROSInterruptException as ros_ex:
        rospy.logfatal("ROS Interrupted.")
        rospy.logfatal(ros_ex)
    except Exception as ex:
        rospy.logfatal("Exception occurred.")
        rospy.logfatal(ex)
    finally:
        rospy.loginfo("Shutting down.")
        rospy.signal_shutdown("Shutting down.")
        sys.exit(0)
