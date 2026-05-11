#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
import os
import math

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians
    pitch is rotation around y in radians
    yaw is rotation around z in radians
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

class LandmarkSaver(Node):
    def __init__(self):
        super().__init__('landmark_saver')
        
        self.declare_parameter('topic_name', '/rtabmap/landmarks')
        self.declare_parameter('output_file', 'saved_landmarks.txt')
        
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.output_file = self.get_parameter('output_file').get_parameter_value().string_value
        
        # Subscribe to the landmark topic
        self.subscription = self.create_subscription(
            PoseArray,
            self.topic_name,
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        
        self.landmarks_dict = {}
        self.get_logger().info(f'Landmark Saver Node Started.')
        self.get_logger().info(f'Subscribed to: {self.topic_name}')
        self.get_logger().info(f'Saving to: {self.output_file}')
        print(f'[INFO] Landmark Saver Node Started. Waiting for messages on {self.topic_name}...')

    def listener_callback(self, msg):
        frame_id = msg.header.frame_id
        num_landmarks = len(msg.poses)
        
        # Immediate terminal log
        print(f'[DEBUG] Received PoseArray message with {num_landmarks} landmarks from frame: {frame_id}')
        
        if num_landmarks == 0:
            return

        updated = False
        # PoseArray doesn't have explicit IDs, so we use the index as ID
        for l_id, pose in enumerate(msg.poses):
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            ox = pose.orientation.x
            oy = pose.orientation.y
            oz = pose.orientation.z
            ow = pose.orientation.w
            
            roll, pitch, yaw = euler_from_quaternion(ox, oy, oz, ow)
            
            # Save or update the landmark
            if l_id not in self.landmarks_dict:
                self.landmarks_dict[l_id] = {'x': x, 'y': y, 'z': z, 'ox': ox, 'oy': oy, 'oz': oz, 'ow': ow, 'roll': roll, 'pitch': pitch, 'yaw': yaw, 'frame_id': frame_id}
                msg_log = f'New Landmark Detected! Index: {l_id} in frame: {frame_id} at ({x:.2f}, {y:.2f}, {z:.2f})'
                self.get_logger().info(msg_log)
                print(f'[INFO] {msg_log}')
                updated = True
            else:
                # Update if position changed significantly
                old = self.landmarks_dict[l_id]
                dist = ((old['x'] - x)**2 + (old['y'] - y)**2 + (old['z'] - z)**2)**0.5
                if dist > 0.05: # 5cm change
                    self.landmarks_dict[l_id] = {'x': x, 'y': y, 'z': z, 'ox': ox, 'oy': oy, 'oz': oz, 'ow': ow, 'roll': roll, 'pitch': pitch, 'yaw': yaw, 'frame_id': frame_id}
                    msg_log = f'Landmark {l_id} Updated! in frame: {frame_id}'
                    self.get_logger().info(msg_log)
                    print(f'[INFO] {msg_log}')
                    updated = True

        if updated:
            self.save_to_file()

    def save_to_file(self):
        try:
            with open(self.output_file, 'w') as f:
                f.write("=== Saved Landmarks ===\n")
                for l_id, data in self.landmarks_dict.items():
                    f.write(f"ID: {l_id} | Frame: {data['frame_id']} | X: {data['x']:.4f} | Y: {data['y']:.4f} | Z: {data['z']:.4f} | oX: {data['ox']:.4f} | oY: {data['oy']:.4f} | oZ: {data['oz']:.4f} | oW: {data['ow']:.4f} | Roll: {data['roll']:.4f} | Pitch: {data['pitch']:.4f} | Yaw: {data['yaw']:.4f}\n")
            print(f'[INFO] [landmark_saver]: Saved {len(self.landmarks_dict)} landmarks to {self.output_file}')
        except Exception as e:
            print(f'[ERROR] [landmark_saver]: Failed to save landmarks: {e}')


def main(args=None):
    rclpy.init(args=args)
    landmark_saver = LandmarkSaver()
    
    try:
        rclpy.spin(landmark_saver)
    except KeyboardInterrupt:
        pass
    finally:
        landmark_saver.save_to_file()
        landmark_saver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
