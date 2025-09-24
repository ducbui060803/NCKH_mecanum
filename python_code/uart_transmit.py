#!/usr/bin/env python3

import rospy
import struct
import serial
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

# Thay đổi cho phù hợp với cổng Arduino của bạn
SERIAL_PORT = '/dev/ttyUSB0'  # hoặc /dev/ttyACM0
BAUD_RATE = 115200
class SerialCommNode:
    def __init__(self):
        rospy.init_node('serial_comm_node')
        print('serial_comm_node')

        # Serial init
        try:
            print("Serial connected to")
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
            print(f"Serial connected to %s", SERIAL_PORT)
            rospy.loginfo("Serial connected to %s", SERIAL_PORT)
        except:
            print("Cannot open serial port!")
            rospy.logerr("Cannot open serial port!")
            exit(1)

        # Subscriber để nhận Twist (vx, vy, omega)
        rospy.Subscriber("/cmd_vel", Twist, self.controller_callback)

        # Publisher để publish V và omega đo được
        self.vel_pub = rospy.Publisher("/uart_data", Float32MultiArray, queue_size=10)

    def controller_callback(self, msg):
        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z

        # Gửi xuống Arduino
        command = f"{vx:.3f} {vy:.3f} {omega:.3f}\n"
        # print(f"Send vx {vx}; vy {vy}; omega {omega}\n")
        self.ser.write(command.encode())

    def run(self):
        rate = rospy.Rate(50)  # 100 Hz
        while not rospy.is_shutdown():
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    parts = line.split()
                    if len(parts) == 4:
                        vx_local_encoder = float(parts[0])
                        vy_local_encoder = float(parts[1])
                        yaw_imu = float(parts[2])
                        yaw_dot = float(parts[3])
                        print(f"Received vx: {vx_local_encoder}; vy: {vy_local_encoder}; yaw_imu: {yaw_imu};  omega: {yaw_dot}\n")
                        msg = Float32MultiArray()
                        msg.data = [vx_local_encoder, vy_local_encoder, yaw_imu, yaw_dot]
                        self.vel_pub.publish(msg)

            except Exception as e:
                rospy.logwarn("Serial error: %s", str(e))
            rate.sleep()

if __name__ == "__main__":
    try:
        node = SerialCommNode()
        node.run()
    except rospy.ROSInterruptException:
        pass