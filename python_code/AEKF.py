#!/usr/bin/env python3

import socket
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
import time

PORT = 5005

def publish_pose(pose):
    position = Float32MultiArray()
    position.data = [pose['x'], pose['y'], pose['phi'], pose['v']]  # Gửi 4 giá trị
    pose_pub.publish(position)


class EKF:
    def __init__(self, dt, Q_fixed, R_fixed):
        """ Khởi tạo EKF với ma trận nhiễu Q và R cố định """
        self.dt = dt  # Bước thời gian (delta_t)

        # Trạng thái hệ thống x_t = [x, y, phi, v]
        self.x_t = np.zeros((4, 1))  

        # Ma trận hiệp phương sai trạng thái P_t
        self.P_t = np.diag([0.1, 0.1, 0.1, 0.1])

        # Ma trận nhiễu quá trình Q_t (Cố định)
        self.Q_t = np.diag(Q_fixed)

        # Ma trận nhiễu đo lường R_t (Cố định)
        self.R_t = np.diag(R_fixed)

        # Ma trận đo lường H_t
        self.H_t = np.array([
            [1, 0, 0, 0],  # X đo từ camera
            [0, 1, 0, 0],  # Y đo từ camera
            [0, 0, 1, 0]   # Phi đo từ camera
        ])
    
    def predict(self, u_t):
        """ Bước dự đoán trạng thái với đầu vào điều khiển u_t = [omega, v] """
        phi_t = self.x_t[2, 0]  # Góc quay hiện tại
        v_t = self.x_t[3, 0]    # Vận tốc hiện tại

        # Ma trận Jacobian F_t
        F_t = np.array([
            [1, 0, -self.dt * v_t * np.sin(phi_t), self.dt * np.cos(phi_t)],
            [0, 1, self.dt * v_t * np.cos(phi_t), self.dt * np.sin(phi_t)],
            [0, 0, 1, self.dt],
            [0, 0, 0, 1]
        ])

        # Ma trận điều khiển B_t
        B_t = np.array([
            [np.cos(phi_t) * self.dt, 0],
            [np.sin(phi_t) * self.dt, 0],
            [0, self.dt],
            [1, 0]
        ])

        # Dự đoán trạng thái mới
        self.x_t = self.x_t + B_t @ u_t  # x_t|t-1 = f(x_t-1, u_t)
        self.P_t = F_t @ self.P_t @ F_t.T + self.Q_t  # Cập nhật ma trận hiệp phương sai P_t|t-1
    
    def update(self, z_t):
        """ Bước cập nhật trạng thái với đo lường mới z_t = [x_meas, y_meas, phi_meas] """
        # Sai số đo lường (Innovation)
        d_t = z_t - self.H_t @ self.x_t  

        # Ma trận hiệp phương sai của Innovation
        S_t = self.H_t @ self.P_t @ self.H_t.T + self.R_t

        # Tính Kalman Gain
        K_t = self.P_t @ self.H_t.T @ np.linalg.inv(S_t)

        # Cập nhật trạng thái
        self.x_t = self.x_t + K_t @ d_t
        self.P_t = (np.eye(4) - K_t @ self.H_t) @ self.P_t

    def get_state(self):
        """ Trả về trạng thái hiện tại """
        return self.x_t

# Khởi tạo EKF với ma trận Q và R cố định
dt = 0.1
Q_fixed = [1e-3, 1e-3, 3e-4, 2e-3]  # Ma trận nhiễu quá trình cố định
R_fixed = [0.5, 0.5, 1]  # Ma trận nhiễu đo lường cố định

ekf = EKF(dt, Q_fixed, R_fixed)

try:
    # Initialize ROS node
    rospy.init_node('pose_estimation_publisher', anonymous=True)

    # ROS publisher for the /odom topic
    pose_pub = rospy.Publisher('/expected_pose', Float32MultiArray, queue_size=10)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))        

    print("Listening for pose data...")

    def shutdown_hook():
        print("Shutting down...")
        sock.close()  # Đóng socket trước khi thoát
    rospy.on_shutdown(shutdown_hook)

    while not rospy.is_shutdown():
        # Nhập dữ liệu từ cảm biến
        theta_dot = 2  
        v = 0.01
        try:
            data, _ = sock.recvfrom(1024)
            x, y, phi = map(float, data.decode().split(","))

            print(f"Received: x={x}, y={y}, phi={phi}")

        except Exception as e:
            print(f"Error processing data: {e}")

        # Nhập dữ liệu đo lường từ Camera + ArUco Marker
        x_meas = x
        y_meas = y
        phi_meas = phi * 3.14 /180  

        # Vector điều khiển và đo lường
        u_t = np.array([[theta_dot], [v]])
        z_t = np.array([[x_meas], [y_meas], [phi_meas]])

        # Bước dự đoán và cập nhật
        ekf.predict(u_t)
        ekf.update(z_t)

        # Lấy trạng thái ước lượng hiện tại
        state = ekf.get_state().flatten()
        print(f"Trạng thái ước lượng: x = {state[0]:.2f}, y = {state[1]:.2f}, phi = {state[2]:.2f}, v = {state[3]:.2f}")
    
        # Store values in the pose dictionary
        pose = {}
        pose['x'] = state[0]
        pose['y'] = state[1]
        pose['phi'] = state[2]
        pose['v'] = state[3]
        position = (float(pose['x']), float(pose['y']), float(pose['phi']) ,float(pose['v']))
        time.sleep(0.5)

        # Publish the position and yaw angle
        publish_pose(pose)
        rospy.loginfo(pose)

except rospy.ROSInterruptException:
    pass

