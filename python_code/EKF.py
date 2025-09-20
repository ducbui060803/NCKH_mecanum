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

v_local = 0
pre_v = 0
acc_prev = 0
yaw_dot = 0
imu_yaw = 0
PORT = 5005

def publish_pose(pose):
    position = Float32MultiArray()
    position.data = [pose['x'], pose['y'], pose['yaw'], pose['v_local'],pose['pose_x'], pose['pose_y'], pose['pose_yaw']]  # Gửi 4 giá trị
    pose_pub.publish(position)

def plot_velocity_compare():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, v_measured_list, 'r-', label="v_local measured (UART)", linewidth=0.8)
    plt.plot(time_stamps, v_estimated_list, 'b--', label="v_local estimated (EKF)", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Comparison of Measured vs Estimated Velocity")
    plt.legend()
    plt.grid(True)
    plt.show()

def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))
class EKF:
    def __init__(self, dt, Q_fixed, R_fixed):
        """ Khởi tạo EKF với ma trận nhiễu Q và R cố định """
        self.dt = dt  # Bước thời gian (delta_t)

        # Trạng thái hệ thống x_t = [x, y, yaw, v_local]
        self.x_t = np.array([
                            [0.0],         # x
                            [0.0],        # y
                            [np.pi/2 ],   # yaw
                            [0.0]          # v_local
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
            [0, 0, 1, 0]   # yaw đo từ camera
        ])
    
    def predict(self, u_t):
        """
        Predict step.
        u_t: [omega_t, delta_v_t]   -- omega from IMU, delta_v from encoder (v_{t} - v_{t-1})
        Model:
          x_{t+1} = x_t + dt * v_t * cos(yaw_t)
          y_{t+1} = y_t + dt * v_t * sin(yaw_t)
          yaw_{t+1} = yaw_t + dt * omega_t
          v_{t+1} = v_t + delta_v_t
        """
        # unpack
        omega = float(u_t[0])
        delta_v = float(u_t[1])

        x = float(self.x_t[0,0])
        y = float(self.x_t[1,0])
        yaw = float(self.x_t[2,0])
        v = float(self.x_t[3,0])

        # --- Nonlinear predict (propagate mean) ---
        dx = v * np.cos(yaw) * self.dt
        dy = v * np.sin(yaw) * self.dt
        dyaw = omega * self.dt
        dv = delta_v  # increment in velocity (paper uses v_{t-1} + Δv)

        # update state
        self.x_t[0,0] = x + dx
        self.x_t[1,0] = y + dy
        self.x_t[2,0] = wrap_to_pi(yaw + dyaw)
        self.x_t[3,0] = v + dv

        # --- Jacobian F_t = df/dx ---
        # partial derivatives evaluated at previous state (using v and yaw)
        F_t = np.array([
            [1., 0., -self.dt * v * np.sin(yaw), self.dt * np.cos(yaw)],
            [0., 1.,  self.dt * v * np.cos(yaw), self.dt * np.sin(yaw)],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])

        # Optionally you can also compute B_t = df/du (control Jacobian) if needed
        # B_t = [[0, dt*cos(yaw)],
        #        [0, dt*sin(yaw)],
        #        [dt, 0],
        #        [0, 1]]

        # --- Covariance propagation ---
        self.P_t = F_t @ self.P_t @ F_t.T + self.Q_t

    def update(self, z_t, H=None, R=None):
        """
        Update step with measurement z_t.
        Default H assumes z = [x_meas, y_meas, yaw_meas].
        If your measurement is different, pass H (matrix) and R (cov) accordingly.
        z_t must be column vector shape (m,1) or 1D array length m.
        """
        # ensure z_t is column vector
        z = np.array(z_t, dtype=float).reshape((-1,1))

        # innovation
        y_tilde = z - H @ self.x_t

        # if yaw measurement present, wrap its error to [-pi,pi]
        # detect yaw row index in H: assume row with [0,0,1,0]
        for i, row in enumerate(H):
            if np.allclose(row, np.array([0.,0.,1.,0.])):
                # wrap difference for yaw measurement
                y_tilde[i,0] = wrap_to_pi(y_tilde[i,0])

        # innovation covariance 
        S = H @ self.P_t @ H.T + R
        
        # Calculate Kalman Gain
        K = self.P_t @ H.T @ np.linalg.inv(S)

        # update state & covariance
        self.x_t = self.x_t + K @ y_tilde
        self.x_t[2,0] = wrap_to_pi(self.x_t[2,0])
        self.P_t = (np.eye(self.P_t.shape[0]) - K @ H) @ self.P_t


    def get_state(self):
        return self.x_t.copy()

    def get_cov(self):
        return self.P_t.copy()


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
        global v_local, imu_yaw, yaw_dot
        if len(msg.data) >= 2:
            v_local = msg.data[0]
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
            x, y, yaw = map(float, data.decode().split(","))

            # print(f"Received: x={x}, y={y}, yaw={yaw}")

        except Exception as e:
            print(f"Error processing data: {e}")

        # Nhập dữ liệu đo lường từ Camera + ArUco Marker
        x_meas = x
        y_meas = y

        if (yaw < 0):
            imu_yaw = -imu_yaw

        yaw_meas = (0.4*yaw + 0.6*imu_yaw) 
        # Vector điều khiển và đo lường
        u_t = np.array([v_local , yaw_dot])

        # Chuẩn hóa góc yaw về [-180:180]
        yaw_meas = np.arctan2(np.sin(yaw_meas), np.cos(yaw_meas))
        z_t = np.array([[x_meas], [y_meas], [yaw_meas]])

        # Bước dự đoán và cập nhật
        ekf.predict(u_t)
        ekf.update(z_t)

        # Lấy trạng thái ước lượng hiện tại
        state = ekf.get_state().flatten()
        pre_v = v_local

        # Store values in the pose dictionary
        pose = {}
        pose['x'] = state[0]
        pose['y'] = state[1]
        pose['yaw'] = state[2]

        pose['v_local'] = state[3]
        pose['pose_x'] = x_meas
        pose['pose_y'] = y_meas
        pose['pose_yaw'] = yaw_meas
        position = (float(pose['x']), float(pose['y']), float(pose['yaw']) ,float(pose['v_local']),float(pose['pose_x']),float(pose['pose_y']))
        # time.sleep(0.01)
        
        # Publish the position and yaw angle
        publish_pose(pose)

        # Lưu giá trị để plot
        now = time.time() - start_time
        time_stamps.append(now)
        v_measured_list.append(v_local)         # v_local từ UART
        v_estimated_list.append(state[3])       # v_local từ EKF
   

    plot_velocity_compare()
except rospy.ROSInterruptException:
    pass

