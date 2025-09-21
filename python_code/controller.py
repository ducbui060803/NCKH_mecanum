#!/usr/bin/env python3

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from scipy.integrate import solve_ivp
import socket
import threading
import json

# Tham số hệ thống
m = 6           # Khối lượng robot (kg)
I_z = 0.22 + 0.02328     # Mô-men quán tính quanh trục z (Xe + banh xe) (kg.m^2)
r = 0.035        # Bán kính bánh xe (m)
L1 = 0       # Khoảng cách từ trọng tâm robot đến tâm robot theo chiều trục x (m)
L2 = 0       # Khoảng cách từ trọng tâm robot đến tâm robot theo chiều trục y (m)

# Ma trận quán tính - Đơn giản hóa mô hình khi L1 = L2 = 0, trọng tâm trùng tâm robot
D = lambda: np.array([
    [m,         0,          -m * L1],
    [0,         m,          m * L2],
    [-m * L1,   m * L2,     I_z + m * (L2**2 + L1**2)]
]) 

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def robot_dynamics(u, state, Ts=0.02):

    # Giải nạp input
    Fx, Fy, Mz = u
    vx, vy, wz = state

    # Nhiễu -- Đơn giản hóa mô hình khi L1 = L2 = 0, trọng tâm trùng tâm robot
    n_zeta = np.array([
        -m * (vy + L2 * wz),
         m * (vx - L1 * wz),
         m * wz * (L2 * vx + L1 * vy)
    ])

    # Gia tốc
    zeta_dot = np.linalg.solve(D(), u - n_zeta)  # [ax, ay, wz_dot]

    # Tích phân Euler để ra vận tốc mới
    vx_local_control = vx + Ts * zeta_dot[0]
    vy_local_control = vy + Ts * zeta_dot[1]
    omega_control = wz + Ts * zeta_dot[2]

    return np.array([vx_local_control, vy_local_control, omega_control])
    
class BacksteppingController:
    # Vector n(ζ) / u = vx, v = vy, r = wz in body frame
    # Đơn giản hóa mô hình khi L1 = L2 = 0, trọng tâm trùng tâm robot
    def n_zeta(self, u, v, r): 
        return np.array([
            -m * (v + L2 * r),
            m * (u - L1 * r),
            m * r * (L2 * u + L1 * v)
        ])
        
    def ekf_callback(self, msg):
        """ Update robot's state from Float32MultiArray. """
        # Lấy giá trị từ message Float32MultiArray
        with self.lock:
            self.x_ekf = msg.data[0]            # pose['x']
            self.y_ekf = msg.data[1]            # pose['y']
            self.yaw_ekf = msg.data[2]          # pose['yaw']
            self.vx_local_enc = msg.data[3]     # vx from EKF measure from encoder data - body frame
            self.vy_local_enc = msg.data[4]     # vy from EKF measure from encoder data - body frame
            self.yaw_dot = msg.data[5]          # yaw rate from IMU - body frame

    def __init__(self):
        rospy.init_node('backstepping_mecanum_controller', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/ekf_pose', Float32MultiArray, self.ekf_callback)

        self.k1 = 8
        self.k2 = 8
        self.plot_initialized = False

        # Global variables updated from socket
        self.x_global_aruco = 0.0           # Send to EKF node
        self.y_global_aruco = 0.0           # Send to EKF node
        self.yaw_aruco = math.pi/2          # Send to EKF node
        self.x_global_desired = 0.0         # Use for controller
        self.y_global_desired = 0.0         # Use for controller
        self.yaw_desired = math.pi/2        # Use for controller
        self.x_dot_global_desired = 0.0     # Use for controller
        self.y_dot_global_desired = 0.0     # Use for controller
        self.x_2dot_global_desired = 0.0    # Use for controller
        self.y_2dot_global_desired = 0.0    # Use for controller
        
        # Global variables from EKF node
        self.x_ekf = 0.0 
        self.y_ekf = 0.0
        self.yaw_ekf = math.pi/2
        self.vx_local_enc = 0.0
        self.vy_local_enc = 0.0
        self.yaw_dot = 0.0  

        # Khởi động socket client trong thread riêng
        self.host = "127.0.0.1"   # IP server (aruco_ui.py)
        self.port = 5005          # Port server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

        self.listener_thread = threading.Thread(target=self.listen_socket, daemon=True)
        self.listener_thread.start()
        # Thông số quỹ đạo
        self.dt = 0.02    # Bước thời gian (s)

        rospy.Timer(rospy.Duration(self.dt), self.control_loop)
        rospy.spin()

    def listen_socket(self):
        # ---------- Thread lắng nghe dữ liệu socket, không chặn control loop ----------
        buffer = ""
        while True:
            try:
                data = self.sock.recv(1024).decode("utf-8")
                if not data:
                    continue
                buffer += data

                # Dữ liệu JSON kết thúc bằng newline
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    try:
                        msg = json.loads(line)
                        self.update_from_socket(msg)
                    except json.JSONDecodeError:
                        rospy.logwarn("JSON lỗi: %s", line)
            except Exception as e:
                rospy.logerr("Lỗi socket: %s", e)
                break

    def update_from_socket(self, msg):
        """Cập nhật biến toàn cục từ JSON."""
        with self.lock:
            self.x_global_aruco = msg.get("x_global_aruco", self.x_global_aruco)
            self.y_global_aruco = msg.get("y_global_aruco", self.y_global_aruco)
            self.yaw_aruco      = msg.get("yaw_aruco", self.yaw_aruco)

            self.x_global_desired       = msg.get("x_global_desired", self.x_global_desired)
            self.y_global_desired       = msg.get("y_global_desired", self.y_global_desired)
            self.yaw_desired            = msg.get("yaw_desired", self.yaw_desired)
            self.x_dot_global_desired   = msg.get("x_dot_global_desired", self.x_dot_global_desired)
            self.y_dot_global_desired   = msg.get("y_dot_global_desired", self.y_dot_global_desired)
            self.yaw_dot_global_desired = 0
            self.x_2dot_global_desired  = msg.get("x_2dot_global_desired", self.x_2dot_global_desired)
            self.y_2dot_global_desired  = msg.get("y_2dot_global_desired", self.y_2dot_global_desired)
            self.yaw_2dot_global_desired = 0
        
    def control_loop(self, event):
        """ Apply backstepping control. """
        with self.lock:
            x_d = self.x_global_desired
            y_d = self.y_global_desired
            yaw_d = self.yaw_desired
            x_dot_d = self.x_dot_global_desired
            y_dot_d = self.y_dot_global_desired
            x_ddot_d = self.x_2dot_global_desired
            y_ddot_d = self.y_2dot_global_desired
            yaw_dot_d = self.yaw_dot_global_desired
            yaw_ddot_d = self.yaw_2dot_global_desired

            x = self.x_ekf
            y = self.y_ekf
            yaw = self.yaw_ekf
            vx_local = self.vx_local_enc
            vy_local = self.vy_local_enc
            yaw_dot_meas = self.yaw_dot
        # e1 is position error in global frame
        e1 = np.array([x_d - x, y_d - y, yaw_d - yaw])
        e1[2] = wrap_to_pi(e1[2])  # Gói góc sai số về [-pi, pi]

        # J is Jacobian matrix to convert from body to global frame
        J = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,                   0,                  1] ])
        
        # J_inv is matrix to convert from global to body frame
        J_inv = np.linalg.pinv(J)

        # eta_dot_d is the desired velocity in the global frame
        eta_dot_d = np.array([x_dot_d, y_dot_d, yaw_dot_d])
        
        # Convert desired velocity from global to body frame
        # eta_dot_d is the desired velocity from trajectory + k1*e1 (position error) in global frame
        zeta_d = np.dot(J_inv, eta_dot_d + self.k1 * e1)

        # Vận tốc đo trong body frame
        zeta_meas_local = np.array([vx_local, vy_local, yaw_dot_meas]) # vx_local_enc, vy_local_enc, yaw_dot should be in body frame

        # Velocity error on body frame
        e2 = zeta_d - zeta_meas_local 

        # Tính vận tốc trong global frame - body to global
        eta_dot_meas = np.dot(J, zeta_meas_local)   # [x_dot, y_dot, yaw_dot] global
        # ============================
        # 3. Derivatives
        # ============================
        
        # e1_dot is the derivative of position error in global frame
        e1_dot = np.array([
            x_dot_d - eta_dot_meas[0],
            y_dot_d - eta_dot_meas[1],
            yaw_dot_d - eta_dot_meas[2]
        ]) # x_dot, y_dot, yaw_dot should be in global frame

        # zeta_dot_d is the desired acceleration in body frame
        # x_ddot_d, y_ddot_d, yaw_ddot_d should be in global frame
        # e1_dot is in global frame
        zeta_dot_d = np.dot(J_inv, np.array([x_ddot_d, y_ddot_d, yaw_ddot_d]) + self.k1 * e1_dot) 


        # D is the inertia matrix
        # k2 is gain for velocity error
        # e2 is velocity error in body frame
        # zeta_dot_d is desired acceleration in body frame
        # n_zeta is disturbance in body frame
        u = np.dot(D(), (self.k2 * e2 + zeta_dot_d + np.dot(J.T, e1))) + self.n_zeta(zeta_meas_local[0], zeta_meas_local[1], zeta_meas_local[2])

        # Saturation (optional)
        u[0:2] = np.clip(u[0:2], -100, 100)
        u[2]   = np.clip(u[2], -20, 20)
        
        vx_control, vy_control, w_control = robot_dynamics(u,[self.vx_local_enc, self.vy_local_enc, self.yaw_dot],
                                                         [self.x_ekf, self.y_ekf, self.yaw_ekf],Ts=0.02)
        # Publish command
        cmd = Twist()

        cmd.linear.x = vx_control
        cmd.linear.y = vy_control
        cmd.angular.z = w_control

        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    try:   
        controller = BacksteppingController()
    except rospy.ROSInterruptException:
        pass

