#!/usr/bin/env python3

import socket
import numpy as np
import math
import rospy
from std_msgs.msg import Float32MultiArray
import time
import matplotlib.pyplot as plt

v_measured_list = []
v_estimated_list = []
time_stamps = []
start_time = time.time()



v = 0
pre_v = 0
acc_prev = 0
yaw_dot = 0
imu_yaw = 0
PORT = 5005

def publish_pose(pose):
    position = Float32MultiArray()
    position.data = [pose['x'], pose['y'], pose['phi'], pose['v'],pose['pose_x'], pose['pose_y'], pose['pose_phi']]  # Gửi 4 giá trị
    pose_pub.publish(position)

def plot_velocity_compare():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, v_measured_list, 'r-', label="V measured (UART)", linewidth=0.8)
    plt.plot(time_stamps, v_estimated_list, 'b--', label="V estimated (EKF)", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Comparison of Measured vs Estimated Velocity")
    plt.legend()
    plt.grid(True)
    plt.show()

class EKF:
    def __init__(self, dt, Q_fixed, R_fixed):
        """ Khởi tạo EKF với ma trận nhiễu Q và R cố định """
        self.dt = dt  # Bước thời gian (delta_t)

        # Trạng thái hệ thống x_t = [x, y, phi, v]
        self.x_t = np.array([
                            [0.0],         # x
                            [0.0],        # y
                            [np.pi/2 ],   # phi
                            [0.0]          # v
                        ])              

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
        global pre_v
        global acc_prev
        """ Bước dự đoán trạng thái với đầu vào điều khiển u_t = [omega, v] """
        phi_t = self.x_t[2, 0]  # Góc quay hiện tại
        v_t = self.x_t[3, 0]    # Vận tốc hiện tại

        # Ma trận Jacobian F_t
        F_t = np.array([
            [1, 0, -self.dt * v_t * np.sin(phi_t), self.dt * np.cos(phi_t)],
            [0, 1, self.dt * v_t * np.cos(phi_t), self.dt * np.sin(phi_t)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        acc = 0.8 * acc_prev + 0.2 * (u_t[0] - pre_v)

        # Ma trận điều khiển B_t
        B_t = np.array([
                [np.cos(phi_t) * dt * u_t[0]],
                [np.sin(phi_t) * dt * u_t[0]],
                [u_t[1]],
                # [u_t[0]-pre_v]
                [acc]
            ])
        # print(f"v_truoc: {self.x_t[3]}")
        # Dự đoán trạng thái mới
        self.x_t = self.x_t + B_t  # x_t|t-1 = f(x_t-1, u_t)
        # print(f"v_sau: {self.x_t[3]}")

        acc_prev = acc
        self.P_t = F_t @ self.P_t @ F_t.T + self.Q_t  # Cập nhật ma trận hiệp phương sai P_t|t-1
        # self.P_t = F_t @ self.P_t @ F_t.T  # Cập nhật ma trận hiệp phương sai P_t|t-1

        self.x_t[2, 0] = np.arctan2(np.sin(self.x_t[2, 0]), np.cos(self.x_t[2, 0]))

        # Predict state and covariance
        # self.x_t = self.x_t + B_t @ u_t.reshape(-1, 1)
        # self.P_t = F_t @ self.P_t @ F_t.T

    def update(self, z_t):
        """ Bước cập nhật trạng thái với đo lường mới z_t = [x_meas, y_meas, phi_meas] """
        # Sai số đo lường (Innovation)
        d_t = z_t - self.H_t @ self.x_t  

        # Ma trận hiệp phương sai của Innovation
        S_t = self.H_t @ self.P_t @ self.H_t.T + self.R_t

        # Tính Kalman Gain
        K_t = self.P_t @ self.H_t.T @ np.linalg.inv(S_t)
        #print(f'K:    {K_t}')
        # print(f'truoc K: {self.x_t[3]}')
        # Cập nhật trạng thái
        self.x_t = self.x_t + K_t @ d_t
        # print(f'sau K: {self.x_t[3]}')
        self.P_t = (np.eye(4) - K_t @ self.H_t) @ self.P_t
        self.x_t[2, 0] = np.arctan2(np.sin(self.x_t[2, 0]), np.cos(self.x_t[2, 0]))


    def get_state(self):
        """ Trả về trạng thái hiện tại """
        return self.x_t

# Khởi tạo EKF với ma trận Q và R cố định
dt = 0.01

Q_fixed = [0.2, 0.2, 0.1, 0.02]
R_fixed = [0.005, 0.005, 0.001]  # Ma trận nhiễu đo lường cố định

ekf = EKF(dt, Q_fixed, R_fixed)
try:
    # Initialize ROS node
    rospy.init_node('pose_estimation_publisher', anonymous=True)

    # ROS publisher for the /odom topic
    pose_pub = rospy.Publisher('/expected_pose', Float32MultiArray, queue_size=10)

    def uart_callback(msg):
        global v, imu_yaw, yaw_dot
        if len(msg.data) >= 2:
            v = msg.data[0]
            imu_yaw = msg.data[1]
            yaw_dot = msg.data[2]

        else:
            rospy.logwarn("UART callback received invalid data!")

    rospy.Subscriber('/measured_vel', Float32MultiArray, uart_callback)
    

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))        

    print("Listening for pose data...")

    def shutdown_hook():
        print("Shutting down...")
        sock.close()  # Đóng socket trước khi thoát
    rospy.on_shutdown(shutdown_hook)

    while not rospy.is_shutdown():
        # Nhập dữ liệu từ cảm biến
        try:
            data, _ = sock.recvfrom(2048)
            x, y, phi = map(float, data.decode().split(","))

            # print(f"Received: x={x}, y={y}, phi={phi}")

        except Exception as e:
            print(f"Error processing data: {e}")

        # Nhập dữ liệu đo lường từ Camera + ArUco Marker
        x_meas = x
        y_meas = y
        # imu_yaw = np.abs(imu_yaw)
        if (phi < 0):
            imu_yaw = -imu_yaw
        # print(f"imu_yaw: {imu_yaw}")
        # print(f"phi:     {phi}")
        # phi_meas = (0.4*phi + 0.6*imu_yaw)  
        phi_meas = (0.2*phi + 0.8*1.57) 
        # Vector điều khiển và đo lường
        u_t = np.array([v , yaw_dot])

        # Chuẩn hóa góc phi về [-180:180]
        phi_meas = np.arctan2(np.sin(phi_meas), np.cos(phi_meas))
        z_t = np.array([[x_meas], [y_meas], [phi_meas]])

        # Bước dự đoán và cập nhật
        ekf.predict(u_t)
        ekf.update(z_t)

        # Lấy trạng thái ước lượng hiện tại
        state = ekf.get_state().flatten()
        pre_v = v
        # print(f"Trạng thái ước lượng: x = {state[0]:.5f}, y = {state[1]:.5f}, phi = {state[2]:.5f}, v = {state[3]:.5f}")
        
        # state[0] = x_meas = x
        # state[1] = y_meas 
        # state[2] = phi_meas 
        # Store values in the pose dictionary
        pose = {}
        pose['x'] = state[0]
        pose['y'] = state[1]
        pose['phi'] = state[2]

        pose['v'] = state[3]
        pose['pose_x'] = x_meas
        pose['pose_y'] = y_meas
        pose['pose_phi'] = phi_meas
        position = (float(pose['x']), float(pose['y']), float(pose['phi']) ,float(pose['v']),float(pose['pose_x']),float(pose['pose_y']))
        # time.sleep(0.01)
        # Publish the position and yaw angle
        publish_pose(pose)

        # Lưu giá trị để plot
        now = time.time() - start_time
        time_stamps.append(now)
        v_measured_list.append(v)         # v từ UART
        v_estimated_list.append(state[3]) # v từ EKF
   
        # rospy.loginfo(pose)

    plot_velocity_compare()
except rospy.ROSInterruptException:
    pass

