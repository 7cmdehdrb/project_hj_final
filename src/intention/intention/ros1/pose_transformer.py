#!/usr/bin/python3

# Basic modules
import os
import sys
import rospy
import roslib
import numpy as np

# TF modules
import tf
from tf.transformations import *


# Messages
from std_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import BoxObject, BoxObjectMultiArray


class PoseTransformer:
    def __init__(self):
        # Global Box on Map
        self.global_box_sub = rospy.Subscriber(
            "/virtual_box", BoxObjectMultiArray, self.global_box_callback
        )
        # Best Box on Base Link
        self.real_best_box_sub = rospy.Subscriber(
            "/best_box/real", BoxObject, self.real_best_box_callback
        )
        self.virtual_best_box_sub = rospy.Subscriber(
            "/best_box/virtual", BoxObject, self.virtual_best_box_callback
        )

        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.global_boxes = BoxObjectMultiArray()
        self.real_best_box = None
        self.virtual_best_box = None

    def real_best_box_callback(self, msg: BoxObject):
        self.real_best_box = msg

    def virtual_best_box_callback(self, msg: BoxObject):
        self.virtual_best_box = msg

    def global_box_callback(self, msg: BoxObjectMultiArray):
        self.global_boxes = msg

    def publish_base_transform(self):
        # Publish map-box
        self.tf_broadcaster.sendTransform(
            translation=(0, 0, 0),
            rotation=(0, 0, 0, 1),
            time=rospy.Time.now(),
            child="box",
            parent="map",
        )

        # Publish box-base_link
        self.tf_broadcaster.sendTransform(
            translation=(-0.0, 0, 0),
            rotation=(0, 0, 0, 1),
            time=rospy.Time.now(),
            child="base_link",
            parent="box",
        )

    def publish_transform(self):
        if self.real_best_box is None or self.virtual_best_box is None:
            # if True:
            # Publish base-transform
            rospy.logwarn("Best box not found.")
            self.publish_base_transform()
            return None

        else:
            # Publish map-box
            # target_global_box = [
            #     global_box
            #     for global_box in self.global_boxes.boxes
            #     if global_box.id == self.real_best_box.id
            # ]

            # if len(target_global_box) == 0:
            #     rospy.logwarn("Best box not found in global boxes.")
            #     self.publish_base_transform()
            #     return

            # target_global_box = target_global_box[0]
            # target_global_box: BoxObject

            target_global_box = self.virtual_best_box

            global_translation = (
                target_global_box.pose.position.x,
                target_global_box.pose.position.y,
                target_global_box.pose.position.z,
            )

            global_rotation = (
                target_global_box.pose.orientation.x,
                target_global_box.pose.orientation.y,
                target_global_box.pose.orientation.z,
                target_global_box.pose.orientation.w,
            )

            # translation = (0, 0, 0)
            global_rotation = (0, 0, 0, 1)

            self.tf_broadcaster.sendTransform(
                translation=global_translation,
                rotation=global_rotation,
                time=rospy.Time.now(),
                child="box",
                parent="map",
            )

            # Publish box-base_link

            local_translation = (
                -self.real_best_box.pose.position.x,
                -self.real_best_box.pose.position.y,
                -self.real_best_box.pose.position.z,
            )

            local_rotation = (
                self.real_best_box.pose.orientation.x,
                self.real_best_box.pose.orientation.y,
                self.real_best_box.pose.orientation.z,
                self.real_best_box.pose.orientation.w,
            )

            local_rotation = (0, 0, 0, 1)

            self.tf_broadcaster.sendTransform(
                translation=local_translation,
                rotation=local_rotation,
                time=rospy.Time.now(),
                child="base_link",
                parent="box",
            )


def main():
    rospy.init_node("pose_transformer")  # TODO: Add node name

    # rospy.spin()
    pose_transformer = PoseTransformer()

    r = rospy.Rate(30)  # TODO: Add rate
    while not rospy.is_shutdown():
        pose_transformer.publish_transform()
        r.sleep()


if __name__ == "__main__":
    main()
    try:
        pass
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
