#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
import time

# --- ROS Node ---

x_aruco = 0.0
y_aruco = 0.0
yaw_aruco = np.pi/2
aruco_detect_flag = 0.0

vx_local = 0.0
vy_local = 0.0
yaw_dot = 0.0
imu_yaw = np.pi/2

path_start_flag = 0.0
# --- EKF class ---
def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

class EKFBody:
    def __init__(self, dt, alpha_Q, alpha_R, Q_fixed, R_fixed):
        self.dt = dt
        self.alpha_Q = alpha_Q  # Hệ số quên cho nhiễu quá trình
        self.alpha_R = alpha_R  # Hệ số quên cho nhiễu đo lường
        # state: x, y, yaw, vx_body, vy_body
        self.x_t = np.array([[0.0], [0.0], [np.pi/2], [0.0], [0.0]])
        self.P_t = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
        self.Q_t = np.diag(Q_fixed)
        self.R_t = np.diag(R_fixed)
        self.H_t = np.array([
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.]
        ])

    def predict(self, u_t):
        x, y, yaw, vx_body, vy_body = self.x_t.flatten()
        omega = float(u_t[0])  # update omega use IMU
        vx_body = float(u_t[1]) # update vx use encoder
        vy_body = float(u_t[2]) # update vy use encoder

        # Fusing yaw với imu
        if (aruco_detect_flag == 1):
            yaw = wrap_to_pi(0.4*yaw + 0.6*imu_yaw)
        else:
            yaw = imu_yaw

        print(f"yaw = {yaw}")
        # --- propagate position & yaw ---
        # Change v_body to v_global
        dx = (vx_body * np.cos(yaw) - vy_body * np.sin(yaw)) * self.dt
        dy = (vx_body * np.sin(yaw) + vy_body * np.cos(yaw)) * self.dt
        dyaw = omega * self.dt

        self.x_t[0,0] = x + dx
        self.x_t[1,0] = y + dy
        self.x_t[2,0] = yaw + dyaw #wrap_to_pi(yaw + dyaw)
        self.x_t[3,0] = vx_body
        self.x_t[4,0] = vy_body
        # vx_body, vy_body giữ nguyên

        # Jacobian
        F_t = np.array([
            [1., 0., (-vx_body*np.sin(yaw) - vy_body*np.cos(yaw))*self.dt, self.dt*np.cos(yaw), -self.dt*np.sin(yaw)],
            [0., 1., (vx_body*np.cos(yaw) - vy_body*np.sin(yaw))*self.dt, self.dt*np.sin(yaw),  self.dt*np.cos(yaw)],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]
        ])
        self.P_t = F_t @ self.P_t @ F_t.T + self.Q_t

    def update(self, z_t):
        """
        Adaptive EKF update step.
        z_t: measurement vector [x, y, phi]
        """
        # Chuyển z_t về dạng cột
        z = np.array(z_t, dtype=float).reshape((-1,1))

        # Sai số đo lường (Innovation)
        y_tilde = z - self.H_t @ self.x_t
        # wrap yaw
        y_tilde[2,0] = wrap_to_pi(y_tilde[2,0])

        # Cập nhật Q_t (adaptive process noise)
        self.Q_t = self.alpha_Q * self.Q_t + (1 - self.alpha_Q) * (self.K_t @ y_tilde @ y_tilde.T @ self.K_t.T)

        # Cập nhật R_t (adaptive measurement noise)
        residual = z - self.H_t @ self.x_t
        self.R_t = self.alpha_R * self.R_t + (1 - self.alpha_R) * (residual @ residual.T + self.H_t @ self.P_t @ self.H_t.T)

        # Innovation covariance
        S = self.H_t @ self.P_t @ self.H_t.T + self.R_t

        # Kalman gain
        K = self.P_t @ self.H_t.T @ np.linalg.inv(S)

        # Cập nhật trạng thái
        self.x_t = self.x_t + K @ y_tilde
        # wrap yaw nếu cần
        # self.x_t[2,0] = wrap_to_pi(self.x_t[2,0])

        # Cập nhật hiệp phương sai
        self.P_t = (np.eye(self.x_t.shape[0]) - K @ self.H_t) @ self.P_t

    def get_state(self):
        return self.x_t.copy()

def uart_callback(msg):
    global vx_local, vy_local, imu_yaw, yaw_dot
    if len(msg.data) >= 4:
        vx_local = msg.data[0]
        vy_local = msg.data[1]
        imu_yaw = msg.data[2]
        yaw_dot = msg.data[3]

def aruco_callback(msg):
    global x_aruco, y_aruco, yaw_aruco, aruco_detect_flag
    if len(msg.data) >= 4:
        x_aruco = msg.data[0]
        y_aruco = msg.data[1]
        yaw_aruco = msg.data[2]
        aruco_detect_flag = msg.data[3]

    # path_start_flag = 1.0 # Start when have aruco marker

def publish_pose(pose):
    msg = Float32MultiArray()
    msg.data = [pose['x_EKF'], pose['y_EKF'], pose['yaw_EKF']]
    pose_pub.publish(msg)

# --- Main ---
dt = 0.01
# Q_fixed = [0.2, 0.2, 0.1, 0.02, 0.02]
# R_fixed = [0.005, 0.005, 0.001]

Q_fixed = [0.03, 0.03, 0.01, 0.005, 0.005] # x~1cm, y~1cm, phi~0.001 rad, v~0.03 m/s
R_fixed = [0.02**2, 0.02**2, np.deg2rad(2)**2] # -> sigma_x = 3 cm, sigma_phi = 1 degree
alpha_Q = 0.9 # Hệ số quên cho nhiễu quá trình
alpha_R = 1 # Hệ số quên cho nhiễu đo lường

ekf = EKFBody(dt, alpha_Q, alpha_R, Q_fixed, R_fixed)

rospy.init_node("ekf_node", anonymous=True)
pose_pub = rospy.Publisher("/ekf_pose", Float32MultiArray, queue_size=10)
rospy.Subscriber("/uart_data", Float32MultiArray, uart_callback)
rospy.Subscriber("/aruco_pose", Float32MultiArray, aruco_callback)

rate = rospy.Rate(100)
time_stamps = []
v_measured_list = []
v_estimated_list = []
start_time = time.time()

while not rospy.is_shutdown():

    # Lấy measurement từ controller node
    x_meas = x_aruco
    y_meas = y_aruco
    yaw_meas = yaw_aruco

    # Chuẩn bị u_t
    u_t = np.array([yaw_dot, vx_local, vy_local])

    # Measurement vector z_t = [x, y, yaw, vx_body, vy_body]
    z_t = [x_meas, y_meas, yaw_meas]

    # EKF
    ekf.predict(u_t)

    if (aruco_detect_flag == 1):
        ekf.update(z_t)

    state = ekf.get_state().flatten()

    # Prepare pose dictionary
    pose = {
        'x_EKF': state[0],
        'y_EKF': state[1],
        'yaw_EKF': state[2],
    }

    print(f"yaw_EKF = {state[2]}")

    publish_pose(pose)

    rate.sleep()