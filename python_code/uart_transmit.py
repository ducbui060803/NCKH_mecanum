#!/usr/bin/env python3
import rospy
import serial
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

# Thay đổi cho phù hợp với cổng Arduino của bạn
SERIAL_PORT = '/dev/ttyUSB0'  # hoặc /dev/ttyACM0
BAUD_RATE = 115200
# V = 0


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
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)

        # Publisher để publish V và omega đo được
        self.vel_pub = rospy.Publisher("/measured_vel", Float32MultiArray, queue_size=10)

    def cmd_vel_callback(self, msg):
        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z

        # Gửi xuống Arduino
        command = f"{vx:.3f} {vy:.3f} {omega:.3f}\n"
        print(f"Send vx {vx}; vy {vy}; omega {omega}\n")
        self.ser.write(command.encode())
    def run(self):
        global V
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            try:
                line = self.ser.readline().decode('utf-8').strip()

                if line:
                    # VD: 0.123 0.045
                    parts = line.split()
                    if len(parts) == 3:
                        V = float(parts[0])
                        # V += 0.01
                        yaw = float(parts[1])
                        theta_dot = float(parts[2])
                        print(f"Received V: {V}; yaw: {yaw};  theta_dot: {theta_dot}\n")
                        msg = Float32MultiArray()
                        msg.data = [V, yaw, theta_dot]
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