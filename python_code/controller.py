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
import time

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

# socket IP
HOST = "0.0.0.0"   # Lắng nghe trên tất cả các interface mạng của IPC
PORT = 5005

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
    
    def publish_pose(self, pose):
        msg = Float32MultiArray()
        msg.data = [pose['x_aruco'], pose['y_aruco'], pose['yaw_aruco'], pose['aruco_detect_flag']]
        self.aruco_pub.publish(msg)

    def ekf_callback(self, msg):
        """ Update robot's state from Float32MultiArray. """
        # Lấy giá trị từ message Float32MultiArray
        with self.lock:
            self.x_ekf = msg.data[0]            # pose['x']
            self.y_ekf = msg.data[1]            # pose['y']
            self.yaw_ekf = msg.data[2]          # pose['yaw']
    
    def uart_callback(self, msg):
        with self.lock:
            self.vx_local_enc = msg.data[0]     # vx from encoder data - body frame
            self.vy_local_enc = msg.data[1]     # vy from encoder data - body frame
            self.yaw_dot = msg.data[3]          # yaw rate from IMU - body frame

    def __init__(self):
        rospy.init_node('backstepping_mecanum_controller', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.aruco_pub = rospy.Publisher('/aruco_pose', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/ekf_pose', Float32MultiArray, self.ekf_callback)
        rospy.Subscriber("/uart_data", Float32MultiArray, self.uart_callback)
        self.k1 = 7
        self.k2 = 6

        self.path_start_flag = 0
        self.stop_flag = 0
        self.robot_state = "WAIT_PATH"  # "WAIT_PATH", "RUNNING", "STOPPED"

        # Global variables updated from socket
        self.x_global_aruco = 0.0           # Send to EKF node
        self.y_global_aruco = 0.0           # Send to EKF node
        self.yaw_aruco = math.pi/2          # Send to EKF node
        self.aruco_detected_flag = 0.0      # Send to EKF node
        self.x_global_desired = 0.0         # Use for controller
        self.y_global_desired = 0.0         # Use for controller
        self.yaw_desired = math.pi/2        # Use for controller
        self.x_dot_global_desired = 0.0     # Use for controller
        self.y_dot_global_desired = 0.0     # Use for controller
        self.yaw_dot_global_desired = 0.0   # Use for controller
        self.x_2dot_global_desired = 0.0    # Use for controller
        self.y_2dot_global_desired = 0.0    # Use for controller
        self.yaw_2dot_global_desired = 0.0  # Use for controller
        
        # Global variables from EKF node
        self.x_ekf = 0.0 
        self.y_ekf = 0.0
        self.yaw_ekf = math.pi/2
        self.vx_local_enc = 0.0
        self.vy_local_enc = 0.0
        self.yaw_dot = 0.0  

        self.lock = threading.Lock()
        # khởi động socket server trong thread riêng
        server_thread = threading.Thread(target=self.start_server, daemon=True)
        server_thread.start()

        # vòng lặp control
        self.dt = 0.02    # Bước thời gian (s)
        rospy.Timer(rospy.Duration(self.dt), self.control_loop)
        rospy.spin()
    
    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        rospy.loginfo(f"Controller server listening on {HOST}:{PORT}")

        while not rospy.is_shutdown():
            conn, addr = server.accept()
            rospy.loginfo(f"Aruco client connected from {addr}")
            client_thread = threading.Thread(target=self.handle_client, args=(conn,), daemon=True)
            client_thread.start()

    def send_to_aruco(self, conn):
        try:
            data = {
                "x_global_EKF": self.x_ekf,
                "y_global_EKF": self.y_ekf,
                "yaw_EKF": self.yaw_ekf
            }
            msg = json.dumps(data) + "\n"
            conn.sendall(msg.encode("utf-8"))
        except Exception as e:
            rospy.logerr(f"Send error: {e}")

    def handle_client(self, conn):
        buffer = ""
        while not rospy.is_shutdown():
            try:
                data = conn.recv(1024).decode("utf-8")
                if not data:
                    break
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    try:
                        msg = json.loads(line)
                        self.update_from_socket(msg)
                        # gửi phản hồi lại Aruco
                        self.send_to_aruco(conn)
                    except json.JSONDecodeError:
                        rospy.logwarn("JSON parse error")
            except Exception as e:
                rospy.logerr(f"Socket error: {e}")
                break
        conn.close()
        rospy.loginfo("Aruco client disconnected")

    def update_from_socket(self, msg):
        """Cập nhật biến toàn cục từ JSON."""
        with self.lock:
            self.x_global_aruco         = msg.get("x_global_aruco", self.x_global_aruco)
            self.y_global_aruco         = msg.get("y_global_aruco", self.y_global_aruco)
            self.yaw_aruco              = msg.get("yaw_aruco", self.yaw_aruco)
            self.aruco_detected_flag    = msg.get("aruco_detected",self.aruco_detected_flag)
            self.x_global_desired       = msg.get("x_global_desired", self.x_global_desired)
            self.y_global_desired       = msg.get("y_global_desired", self.y_global_desired)
            self.yaw_desired            = msg.get("yaw_desired", self.yaw_desired)
            self.x_dot_global_desired   = msg.get("x_dot_global_desired", self.x_dot_global_desired)
            self.y_dot_global_desired   = msg.get("y_dot_global_desired", self.y_dot_global_desired)
            self.yaw_dot_global_desired = msg.get("yaw_dot_global_desired", self.yaw_dot_global_desired)
            self.x_2dot_global_desired  = msg.get("x_2dot_global_desired", self.x_2dot_global_desired)
            self.y_2dot_global_desired  = msg.get("y_2dot_global_desired", self.y_2dot_global_desired)
            self.yaw_2dot_global_desired = msg.get("yaw_2dot_global_desired", self.yaw_2dot_global_desired)

            # Nhận path = bật flag
            self.path_start_flag         = msg.get("path_start_flag", self.path_start_flag)
            # Nhận lệnh stop
            self.stop_flag               = msg.get("stop_flag", self.stop_flag)

    def control_loop(self, event):
        """ Apply backstepping control. """
                # Xác định trạng thái
        if self.stop_flag:
            self.robot_state = "STOPPED"
        elif self.path_start_flag:
            self.robot_state = "RUNNING"
        else:
            self.robot_state = "WAIT_PATH"

        # WAIT_PATH hoặc STOPPED → dừng robot
        if self.robot_state != "RUNNING":
            cmd = Twist()
            cmd.linear.x = 0
            cmd.linear.y = 0
            cmd.angular.z = 0
            self.cmd_pub.publish(cmd)
            return

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
        
        vx_control, vy_control, w_control = robot_dynamics(u,[self.vx_local_enc, self.vy_local_enc, self.yaw_dot], Ts=0.02)
        # Publish command
        cmd = Twist()

        cmd.linear.x = vx_control
        cmd.linear.y = vy_control
        cmd.angular.z = w_control

        self.cmd_pub.publish(cmd)

        
        print(f"vx_control ={vx_control}, vy_control = {vy_control}, w_control = {w_control}")

        # Prepare pose dictionary
        aruco_pose = {
            'x_aruco': self.x_global_aruco,
            'y_aruco': self.y_global_aruco,
            'yaw_aruco': self.yaw_aruco,
            'aruco_detect_flag' : self.aruco_detected_flag,
        }

        self.publish_pose(aruco_pose)

if __name__ == '__main__':
    try:   
        controller = BacksteppingController()
    except rospy.ROSInterruptException:
        pass

