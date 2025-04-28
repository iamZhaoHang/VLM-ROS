#!/usr/bin/env python3

import random
import rospy
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import threading


class RobotControlNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("robot_control_node", anonymous=True)

        # State variables
        self.awake = False  # Robot is in awake mode or not
        self.current_angle = 0.0  # Current yaw angle from odom
        self.awake_timer = None  # Timer for exiting awake mode
        self.last_keyword_time = rospy.Time.now()  # Last time a keyword was received

        # Variables for distance control
        self.start_position = None  # 起始位置 (x, y)
        self.target_distance = 0.0  # 目标行驶距离
        self.moving_forward = False # 标记是否正在前进
        self.moving_backward = False # 标记是否正在后退

        # Publishers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.input_text_pub = rospy.Publisher("/tts_input", String, queue_size=10)  # For status messages

        # Subscribers
        rospy.Subscriber("/keyword_result", String, self.keyword_result_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/awake_angle", Float32, self.awake_angle_callback)

        # Initialize command state
        self.current_twist = Twist()
        self.stop_publishing = False

        # Start the background thread for publishing cmd_vel
        self.cmd_vel_thread = threading.Thread(target=self.publish_cmd_vel)
        self.cmd_vel_thread.daemon = True
        self.cmd_vel_thread.start()

        rospy.loginfo("Robot Control Node started.")

    def odom_callback(self, msg):
        """Callback to handle odometry messages."""
        # Extract position and yaw angle from the odometry message
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y**2 + orientation_q.z**2)
        self.current_angle = math.atan2(siny_cosp, cosy_cosp)

        current_x = position.x
        current_y = position.y

        # Distance control logic
        if self.start_position is not None and (self.moving_forward or self.moving_backward) :
            start_x, start_y = self.start_position
            distance_traveled = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2) # 欧氏距离

            if distance_traveled >= self.target_distance:
                rospy.loginfo(f"目标距离 {self.target_distance:.2f} 米已到达，停止移动.")
                self.current_twist.linear.x = 0.0  # 停止线速度
                self.cmd_vel_pub.publish(self.current_twist)
                self.start_position = None  # 重置起始位置
                self.target_distance = 0.0   # 重置目标距离
                self.moving_forward = False
                self.moving_backward = False
                if self.moving_forward:
                    self.publish_input_text("指令完成")
                elif self.moving_backward:
                    self.publish_input_text("指令完成")

        # Update current position (you might need to store current_x, current_y if needed elsewhere)
        # self.current_position = (current_x, current_y) # 可选，如果其他地方需要当前位置信息
    

    def awake_angle_callback(self, msg):
        """Callback to handle awake_angle messages."""
        rospy.loginfo(f"Received awake angle: {msg.data}")
        self.awake = True  # Enter awake mode
        self.last_keyword_time = rospy.Time.now()  # Reset the timer
        self.publish_input_text(f"进入唤醒状态，角度: {msg.data}")

    def keyword_result_callback(self, msg):
        """Callback to handle keyword result messages."""
        recognition_result = msg.data.strip()  # Get the command string
        rospy.loginfo(f"Received keyword result: {recognition_result}")

        # Update the awake timer
        self.last_keyword_time = rospy.Time.now()

        # Check for wake-up command or handle commands if in awake mode
        if recognition_result == "你好小派":
            rospy.loginfo("Entering awake mode.")
            self.awake = True  # Enter awake mode
            self.publish_input_text("我在！")
        elif self.awake:
            # Process commands in awake mode
            self.handle_command(recognition_result)

    def handle_command(self, command):
        rospy.loginfo("""Handle robot control commands.""")
        response_message = None  # To store the response message for input_text

        if command == "前进":
            rospy.loginfo("Command: Move forward")
            self.current_twist.linear.x = 0.2  # Forward linear velocity
            response_message = "正在前进"
        elif command == "后退":
            rospy.loginfo("Command: Move backward")
            self.current_twist.linear.x = -0.15  # Backward linear velocity
            response_message = "正在后退"
        elif command == "左转":
            rospy.loginfo("Command: Turn left")
            self.current_twist.angular.z = 0.3  # Left angular velocity
            response_message = "正在左转"
        elif command == "右转":
            rospy.loginfo("Command: Turn right")
            self.current_twist.angular.z = -0.3  # Right angular velocity
            response_message = "正在右转"
        elif command == "回中" or command == "直行":
            rospy.loginfo("Command: Center or move straight")
            self.current_twist.angular.z = 0.0  # Stop turning
            response_message = "正在回中"
        elif command == "停止":
            rospy.loginfo("Command: Stop")
            self.current_twist.linear.x = 0.0
            self.current_twist.angular.z = 0.0  # Stop all movement
            self.start_position = None # 重置距离控制相关变量
            self.target_distance = 0.0
            self.moving_forward = False
            self.moving_backward = False
            response_message = "已停止"
        elif command == "转个圈":
            rospy.loginfo("Command: Spin around")
            response_message = "正在转圈"
            self.publish_input_text(response_message)
            self.spin_in_circle()
            return
        elif command == "过来":
            rospy.loginfo("Command: Come here")
            response_message = "正在过来"
            # TODO: Implement the logic for "Come here" command
        elif command == "向前一米":
            rospy.loginfo("Command: Move forward 1 meter")
            if msg := rospy.wait_for_message("/odom", Odometry, timeout=5): # 获取当前odom信息，防止start_position为None
                position = msg.pose.pose.position
                self.start_position = (position.x, position.y) # 记录起始位置
                self.target_distance = 1.0  # 设置目标距离为 1 米
                self.current_twist.linear.x = 0.2  # 设置前进速度 (可以调整)
                self.moving_forward = True
                self.moving_backward = False
                response_message = "正在向前一米"
            else:
                rospy.logwarn("No odom message received, cannot start '前进一米' command.")
                response_message = "无法获取里程计信息"
        elif command == "向后一米":
            rospy.loginfo("Command: Move backward 1 meter")
            if msg := rospy.wait_for_message("/odom", Odometry, timeout=5): # 获取当前odom信息，防止start_position为None
                position = msg.pose.pose.position
                self.start_position = (position.x, position.y) # 记录起始位置
                self.target_distance = 1.0  # 设置目标距离为 1 米
                self.current_twist.linear.x = -0.15 # 设置后退速度 (可以调整)
                self.moving_forward = False
                self.moving_backward = True
                response_message = "正在向后一米"
            else:
                rospy.logwarn("No odom message received, cannot start '后退一米' command.")
                response_message = "无法获取里程计信息"
        else:
            rospy.logwarn(f"Unrecognized command: {command}")
            response_message = f"无法识别的指令：{command}"

        # Publish the response message via input_text
        if response_message:
            self.publish_input_text(response_message)


    def spin_in_circle(self):
        """Make the robot perform a 360-degree circle at a fixed radius."""
        rospy.loginfo("Executing spin in circle...")
        radius = 0.5  # Circle radius in meters
        linear_speed = 0.2  # Linear velocity in meters per second
        angular_speed = linear_speed / radius  # Angular velocity (v = r * w)

        # Calculate the target angle (360 degrees from the current angle)
        start_angle = self.current_angle
        target_angle = start_angle + 2 * math.pi

        # Publish velocity commands to make the robot spin
        rate = rospy.Rate(10)  # 10 Hz
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed

        while self.current_angle < target_angle:
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Stop the robot after completing the circle
        self.current_twist.linear.x = 0.0
        self.current_twist.angular.z = 0.0
        rospy.loginfo("Spin in circle completed.")

    def publish_input_text(self, message):
        """Publish the input text to a ROS topic."""
        msg = String()
        msg.data = message
        self.input_text_pub.publish(msg)
        rospy.loginfo(f"Published input text: {message}")

    def publish_cmd_vel(self):
        """Continuously publish cmd_vel at a frequency of at least 10 Hz."""
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.awake:
                # Exit awake mode if no commands received in the last 20 seconds
                if (rospy.Time.now() - self.last_keyword_time).to_sec() > 200:
                    rospy.loginfo("Exiting awake mode due to timeout.")
                    self.awake = False
                    self.current_twist.linear.x = 0.0
                    self.current_twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(self.current_twist)
                    self.publish_input_text("我先退下啦!")
                # Publish the current twist
                
                self.cmd_vel_pub.publish(self.current_twist)
            rate.sleep()

    def run(self):
        """Run the main loop."""
        rospy.spin()


if __name__ == "__main__":
    try:
        node = RobotControlNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Robot Control Node.")
