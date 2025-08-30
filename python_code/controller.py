#!/usr/bin/env python3

import rospy
import numpy as np
from matplotlib.ticker import MultipleLocator

import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from scipy.integrate import solve_ivp
import queue
import matplotlib.pyplot as plt
import threading
import keyboard


threshold = 0.1
input_data_queue = queue.LifoQueue()

# Tham số hệ thống
m = 6           # Khối lượng robot (kg)
I_z = 0.22 + 0.02328     # Mô-men quán tính quanh trục z (Xe + banh xe) (kg.m^2)
r = 0.035        # Bán kính bánh xe (m)
L1 = 0.11       # Khoảng cách từ tâm đến bánh xe (m)
L2 = 0.1       # Khoảng cách từ tâm đến bánh xe (m)

# Ma trận quán tính
D = lambda: np.array([
    [m,         0,          -m * L1],
    [0,         m,          m * L2],
    [-m * L1,   m * L2,     I_z + m * (L2**2 + L1**2)]
]) 
alpha = 0.7
a_a = 0.3
filtered_x = None
filtered_y = None
filtered_angle = None
def LowPassFilter(x, y, angle):
    global filtered_x, filtered_y, filtered_angle,alpha, a_a
    if filtered_x is None or filtered_y is None or filtered_angle is None:
        filtered_x = x
        filtered_y = y
        filtered_angle = angle
    else:
        filtered_x = alpha * x + (1 - alpha) * filtered_x
        filtered_y = alpha * y + (1 - alpha) * filtered_y
        filtered_angle = a_a * angle + (1 - a_a) * filtered_angle
    return filtered_x, filtered_y, filtered_angle

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# def generate_capsule_path(radius, height, points_per_arc=50, points_per_line=50):
#     # Góc cho nửa tròn
#     yaw_top = np.linspace(np.pi, 0, points_per_arc)        # từ trái sang phải (nửa trên)
#     yaw_bottom = np.linspace(0, -np.pi, points_per_arc)    # từ phải sang trái (nửa dưới)

#     # Nửa tròn trên (trái -> phải)
#     top_arc_x = radius * np.cos(yaw_top)
#     top_arc_y = radius * np.sin(yaw_top) + height / 2

#     # Đoạn thẳng bên phải (trên -> dưới)
#     # right_line_x = np.linspace(radius, radius, points_per_line)
#     # right_line_y = np.linspace(height / 2, -height / 2, points_per_line)

#     # # Nửa tròn dưới (phải -> trái)
#     # bottom_arc_x = radius * np.cos(yaw_bottom)
#     # bottom_arc_y = radius * np.sin(yaw_bottom) - height / 2

#     # # Đoạn thẳng bên trái (dưới -> trên)
#     # left_line_x = np.linspace(-radius, -radius, points_per_line)
#     # left_line_y = np.linspace(-height / 2, height / 2, points_per_line)

#     # Ghép các đoạn lại theo thứ tự liên tục
#     x_traject  = np.concatenate([top_arc_x])
#     y_traject  = np.concatenate([top_arc_y])
    
#     # x_traject  = np.concatenate([left_line_x])
#     # y_traject  = np.concatenate([left_line_y])
#         # Tính góc theo trục X cho mỗi điểm (trừ điểm đầu và cuối)
#     angles_traject = []
#     for i in range(1, len(x_traject) - 1):
#         dx = x_traject[i + 1] - x_traject[i - 1]
#         dy = y_traject[i + 1] - y_traject[i - 1]
#         angle = np.arctan2(dy, dx)  # Tính góc theo trục X
#         angles_traject.append(angle)

#     # Thêm góc cho điểm đầu và cuối (lấy từ điểm tiếp theo và trước đó)
#     angles_traject = [angles_traject[0]] + angles_traject + [angles_traject[-1]]
#     return x_traject, y_traject, angles_traject

def generate_path(path_type="circle", length=1.0, radius=1.0, points=100, angle_fixed=np.pi/2, T=10.0):
    """
    path_type   : "circle", "line_x", "line_y", "square", "figure8"
    length      : độ dài đường thẳng hoặc cạnh hình vuông
    radius      : bán kính hình tròn hoặc hình số 8
    points      : số điểm trên quỹ đạo
    angle_fixed : góc cố định của robot (rad)
    T           : tổng thời gian thực hiện quỹ đạo (s)
    """
    t = np.linspace(0, T, points)
    dt = t[1] - t[0]

    if path_type == "circle":
        omega = 2 * np.pi / T
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)

        x_dot = -radius * omega * np.sin(omega * t)
        y_dot =  radius * omega * np.cos(omega * t)

        x_ddot = -radius * omega**2 * np.cos(omega * t)
        y_ddot = -radius * omega**2 * np.sin(omega * t)

    elif path_type == "line_x":
        v = length / T
        x = v * t
        y = np.zeros_like(t)

        x_dot = np.full_like(t, v)
        y_dot = np.zeros_like(t)

        x_ddot = np.zeros_like(t)
        y_ddot = np.zeros_like(t)

    elif path_type == "line_y":
        v = length / T
        x = np.zeros_like(t)
        y = v * t

        x_dot = np.zeros_like(t)
        y_dot = np.full_like(t, v)

        x_ddot = np.zeros_like(t)
        y_ddot = np.zeros_like(t)

    elif path_type == "square":
        # Ở đây không có công thức giải tích -> dùng sai phân số
        side_points = points // 4
        x = []
        y = []
        # cạnh 1
        x.extend(np.linspace(0, length, side_points))
        y.extend(np.zeros(side_points))
        # cạnh 2
        x.extend(np.full(side_points, length))
        y.extend(np.linspace(0, length, side_points))
        # cạnh 3
        x.extend(np.linspace(length, 0, side_points))
        y.extend(np.full(side_points, length))
        # cạnh 4
        x.extend(np.zeros(side_points))
        y.extend(np.linspace(length, 0, side_points))

        x = np.array(x)
        y = np.array(y)

        # Đạo hàm bằng sai phân số
        x_dot = np.gradient(x, dt)
        y_dot = np.gradient(y, dt)
        x_ddot = np.gradient(x_dot, dt)
        y_ddot = np.gradient(y_dot, dt)

    elif path_type == "figure8":
        omega = 2 * np.pi / T
        x = radius * np.sin(omega * t)
        y = radius * np.sin(omega * t) * np.cos(omega * t)

        x_dot = radius * omega * np.cos(omega * t)
        y_dot = radius * omega * (np.cos(2 * omega * t))

        x_ddot = -radius * omega**2 * np.sin(omega * t)
        y_ddot = -2 * radius * omega**2 * np.sin(2 * omega * t)

    else:
        raise ValueError("Unknown path_type")

    # Góc robot giữ nguyên
    yaw = np.full_like(t, angle_fixed)
    yaw_dot = np.zeros_like(t)
    yaw_ddot = np.zeros_like(t)

    return x, y, yaw, x_dot, y_dot, yaw_dot, x_ddot, y_ddot, yaw_ddot

def robot_dynamics(u, state, pose, Ts=0.01):
    # Thông số hệ thống
    m = 6
    L1 = 0.1
    L2 = 0.1
    Iz = 0.22 + 0.02328

    # Giải nạp input
    Fx, Fy, Mz = u
    vx, vy, wz = state
    x, y, theta = pose

    # Ma trận quán tính
    D = np.array([
        [m,   0,   -m*L1],
        [0,   m,    m*L2],
        [-m*L1, m*L2, Iz + m*(L1**2 + L2**2)]
    ])

    # Nhiễu
    n_zeta = np.array([
        -m * (vy + L2 * wz),
         m * (vx - L1 * wz),
         m * wz * (L2 * vx + L1 * vy)
    ])

    # Gia tốc
    zeta_dot = np.linalg.solve(D, u - n_zeta)  # [ax, ay, wz_dot]

    # Tích phân Euler để ra vận tốc mới
    vx_next = vx + Ts * zeta_dot[0]
    vy_next = vy + Ts * zeta_dot[1]
    wz_next = wz + Ts * zeta_dot[2]

    # Kinematic (body -> global)
    x_dot = np.cos(theta)*vx - np.sin(theta)*vy
    y_dot = np.sin(theta)*vx + np.cos(theta)*vy
    theta_dot = wz

    x_next = x + Ts * x_dot
    y_next = y + Ts * y_dot
    theta_next = np.arctan2(np.sin(theta + Ts * theta_dot), np.cos(theta + Ts * theta_dot))  # wrap [-pi,pi]

    return np.array([x_next, y_next, theta_next, vx_next, vy_next, wz_next])
    
class BacksteppingController:
    # Vector n(ζ)
    def n_zeta(self, u, v, r):
        return np.array([
            -m * (v + L2 * r),
            m * (u - L1 * r),
            m * r * (L2 * u + L1 * v)
        ])
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0
        cmd.linear.y = 0
        cmd.angular.z = 0
        self.cmd_pub.publish(cmd)

    # def keyboard_listener(self):
    #     keyboard.wait('q')
    #     rospy.loginfo("Stopping robot due to 'q' key press.")
    #     self.running = False
    #     for i in range(0,3):    
    #         self.stop_robot()
    #     rospy.signal_shutdown("Stopped by user.")

    def odom_callback(self, msg):
        """ Update robot's state from Float32MultiArray. """
        # Lấy giá trị từ message Float32MultiArray
        if (self.trajectory_index < self.N):
            self.x = msg.data[0]  # pose['x']
            self.y = msg.data[1]  # pose['y']
            self.yaw = msg.data[2]  # pose['phi']
            self.v = msg.data[3]
            self.pose_x = msg.data[4]
            self.pose_y = msg.data[5]
            self.pose_phi = msg.data[6]
            # print(f"v: {self.v} ")
            self.x_dot = self.v * math.cos(self.yaw)
            self.y_dot = self.v * math.sin(self.yaw) 
        else:
            self.x = 0
            self.y = 0
            self.yaw = 0
            self.v = 0
            self.pose_x = 0
            self.pose_y = 0
            self.pose_phi = 0
            self.x_dot = 0
            self.y_dot = 0 
    
    def uart_callback(self, msg):
        if (self.trajectory_index < self.N):
            self.yaw_dot = msg.data[2]
        else:
            self.yaw_dot = 0

    def __init__(self):
        rospy.init_node('backstepping_mecanum_controller', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/expected_pose', Float32MultiArray, self.odom_callback)

        rospy.Subscriber('/measured_vel', Float32MultiArray, self.uart_callback)
        # Controller gains AEKF
        # self.k1 = 5
        # self.k2 = 0.7
        # self.k1 = 13
        # self.k2 = 1
        # Controller gains EKF
        # self.k1 = 12
        # self.k2 = 0.9
        # Controller gains chi
        # self.k1 = 15
        # self.k2 = 0.8

        self.k1 = 4
        self.k2 = 0.1
        self.plot_initialized = False

        # Robot state
        self.x = 0.0 
        self.y = 0.0
        self.yaw = math.pi/2
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.yaw_dot = 0.0
        self.v = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_phi = math.pi/2
        # Thông số quỹ đạoq
        # self.T = 32.0       # Tổng thời gian (s)
        # self.T = 40.0       # Tổng thời gian (s)
        self.T = 10.0       # Tổng thời gian (s)

        self.dt = 0.01     # Bước thời gian (s)
        self.N = int(self.T / self.dt)
        self.trajectory_index = 0
        
        # self.x_traject, self.y_traject, self.angles_traject = generate_circle_path(1 ,self.N)
        self.pathtype = "line_x"
        self.x_traject, self.y_traject, self.angles_traject, \
        self.x_dot_traject, self.y_dot_traject, self.yaw_dot_traject, \
        self.x_ddot_traject, self.y_ddot_traject, self.yaw_ddot_traject = generate_path(self.pathtype, 1.0, 1.0, self.N, np.pi/2, self.T)

        self.path_points = list(zip(self.x_traject, self.y_traject, self.angles_traject))
        self.velocity_points = list(zip(self.x_dot_traject, self.y_dot_traject, self.yaw_dot_traject))
        self.accelerations_points = list(zip(self.x_ddot_traject, self.y_ddot_traject, self.yaw_ddot_traject))
        # Ve quy dao 
        self.error_x = []
        self.error_y = []
        self.actual_vx = []
        self.actual_vy = []
        self.desired_vx = []
        self.desired_vy = []
        self.error_velo_x = []
        self.error_velo_y = []
        self.error_time = []
        self.actual_path = []
        self.desired_path = [] 
        self.mearsure_path = []
        self.u_history = []
        # Stop variable
        # self.running = True
        # threading.Thread(target=self.keyboard_listener, daemon=True).start()
        # Tạo quỹ đạo tham chiếu (linear interpolation)
        # self.x_traj = np.linspace( -0.35,  -0.35, self.N)  # x không đổi
        # self.y_traj = np.linspace(-1.4, 0, self.N) # y từ -1 lên 0
        # self.yaw_traj = np.linspace(np.pi/2, np.pi/2, self.N)  # không đổi hướng
        self.x_traj = [pt[0] for pt in self.path_points]
        self.y_traj = [pt[1] for pt in self.path_points]
        self.yaw_traj = [pt[2] for pt in self.path_points]
        self.x_dot_traj = [pt[0] for pt in self.velocity_points]
        self.y_dot_traj = [pt[1] for pt in self.velocity_points]
        self.yaw_dot_traj = [pt[2] for pt in self.velocity_points]
        self.x_ddot_traj = [pt[0] for pt in self.accelerations_points]
        self.y_ddot_traj = [pt[1] for pt in self.accelerations_points]
        self.yaw_ddot_traj = [pt[2] for pt in self.accelerations_points]
        # Gọi timer để cập nhật trajectory mỗi dt giây
        rospy.Timer(rospy.Duration(self.dt), self.update_trajectory)

        rospy.Timer(rospy.Duration(self.dt), self.control_loop)
        rospy.spin()
    
    def update_trajectory(self, event):
        if self.trajectory_index < self.N-1:
            # print(f"idex: {self.trajectory_index} and N {self.N}")
            self.x_d = self.x_traj[self.trajectory_index]
            self.y_d = self.y_traj[self.trajectory_index]
            self.yaw_d = self.yaw_traj[self.trajectory_index]
            self.x_dot_d = self.x_dot_traj[self.trajectory_index]
            self.y_dot_d = self.y_dot_traj[self.trajectory_index]
            self.yaw_dot_d = self.yaw_dot_traj[self.trajectory_index]
            self.x_ddot_d = self.x_ddot_traj[self.trajectory_index]
            self.y_ddot_d = self.y_ddot_traj[self.trajectory_index]
            self.yaw_ddot_d = self.yaw_ddot_traj[self.trajectory_index]
            self.trajectory_index += 1
        else:
            # Giữ trạng thái cuối cùng nếu đã vượt quá quỹ đạo
            self.x = self.x_d
            self.y = self.y_d
            self.yaw = self.yaw_d

    def control_loop(self, event):
        """ Apply backstepping control. """

        # ============================
        # 1. Desired velocities (x_dot_d, y_dot_d, yaw_dot_d)
        # ============================
        # self.x_dot_d = -r * self.omega * np.sin(self.yaw_d * event.current_real.to_sec())
        # self.y_dot_d =  r * self.omega * np.cos(self.yaw_d * event.current_real.to_sec())
        # self.yaw_dot_d = self.omega

        # Desired accelerations (cần nếu tính zeta_dot_d)
        # self.x_ddot_d = -r * self.omega**2 * np.cos(self.yaw_d * event.current_real.to_sec())
        # self.y_ddot_d = -r * self.omega**2 * np.sin(self.yaw_d * event.current_real.to_sec())
        # self.yaw_ddot_d = 0.0

        # ============================
        # 2. Errors
        # ============================
        self.x = 0
        self.yaw = np.pi/2
        e1 = np.array([self.x_d - self.x, self.y_d - self.y, self.yaw_d - self.yaw])
        e1[2] = wrap_to_pi(e1[2])  # Gói góc sai số về [-pi, pi]

        # Jacobian
        J = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw),  np.cos(self.yaw), 0],
            [0,                   0,                  1] ])
        J_inv = np.linalg.pinv(J)

        # Reference velocity (zeta_d)
        eta_dot_d = np.array([self.x_dot_d, self.y_dot_d, self.yaw_dot_d])
        zeta_d = np.dot(J_inv, eta_dot_d + self.k1 * e1)

        # Velocity error
        e2 = zeta_d - np.array([self.x_dot, self.y_dot, self.yaw_dot])

        # ============================
        # 3. Derivatives
        # ============================
        e1_dot = np.array([
            self.x_dot_d - self.x_dot,
            self.y_dot_d - self.y_dot,
            self.yaw_dot_d - self.yaw_dot
        ])

        zeta_dot_d = np.dot(J_inv, np.array([
            self.x_ddot_d,
            self.y_ddot_d,
            self.yaw_ddot_d
        ]) + self.k1 * e1_dot)

        u = np.dot(D(), (self.k2 * e2 + zeta_dot_d + np.dot(J.T, e1))) + self.n_zeta(self.x_dot, self.y_dot, self.yaw_dot)

        # Saturation (optional)
        u[0:2] = np.clip(u[0:2], -100, 100)
        u[2]   = np.clip(u[2], -20, 20)
        
        x_next, y_next, theta_next, vx_next, vy_next, wz_next = robot_dynamics(u,[self.x_dot, self.y_dot, self.yaw_dot],[self.x, self.y, self.yaw],Ts=0.01)
        # Publish command
        cmd = Twist()
        cmd.linear.x = vx_next
        cmd.linear.y = vy_next
        cmd.angular.z = wz_next

        self.cmd_pub.publish(cmd)

        # Draw path
        # self.u_history.append([u[0], u[1], u[2]])
        self.u_history.append([vx_next, vy_next, wz_next])
        self.mearsure_path.append((self.x, self.y, self.yaw))
        self.actual_path.append((self.pose_x, self.pose_y, self.pose_phi))
        self.desired_path.append((self.x_d, self.y_d, self.yaw_d))

        self.actual_vx.append(self.x_dot)
        self.actual_vy.append(self.y_dot)
        self.desired_vx.append(self.x_dot_d)
        self.desired_vy.append(self.y_dot_d)
        
        self.error_x.append(abs(self.x_d - self.pose_x))
        self.error_y.append(abs(self.y_d - self.pose_y))
        self.error_velo_x.append(abs(self.x_dot_d - self.x_dot))
        self.error_velo_y.append(abs(self.y_dot_d - self.y_dot))
        self.error_time.append(self.trajectory_index * self.dt)  # hoặc rospy.get_time() nếu cần chính xác tuyệt đối
        # print(f"x_a = {self.pose_x}, y_a = {self.pose_y}")
        # print(f"x_m = {self.x}, y_m = {self.y}, a_m = {self.yaw}")
        # print(f"x_d = {self.x_d}, y_d = {self.y_d}, a_d = {self.yaw_d}")

def plot_paths(desired, actual, mearsure):
    desired  = np.array(desired)
    actual   = np.array(actual)
    mearsure = np.array(mearsure)

    plt.figure(figsize = (8, 6))
    # plt.plot(desired[:, 0], desired[:, 1], 'r-', linewidth = 0.5, label='Desired Path')
    # plt.plot(mearsure[:, 0], mearsure[:, 1], 'b-', linewidth = 1, label='Mearsure - Aruko Path')
    # plt.plot(actual[:, 0], actual[:, 1], 'g-', linewidth = 0.6, label='Actual - xEKF Path')
    plt.plot(desired[:, 0], desired[:, 1], 'r-', linewidth = 0.5, label='Desired Path')
    # plt.plot(mearsure[:, 0], mearsure[:, 1], 'b-', linewidth = 1, label='Mearsure - Aruko Path')
    plt.plot(actual[:, 0], actual[:, 1], 'g-', linewidth = 0.6, label='Actual - xEKF Path')

    # plt.scatter(mearsure[:, 0], mearsure[:, 1], c='b--', s=5, label='Mearsure Path')  # Dấu chấm nhỏ
    # plt.scatter(actual[:, 0], actual[:, 1], c='g', s=3, label='Actual Path')  # Dấu chấm nhỏ
    plt.title("Robot Path Tracking")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid(False)
    plt.show()

def plot_position_vs_reference(controller):
    time_stamps   = controller.error_time
    x_actual      = [pos[0] for pos in controller.mearsure_path]
    y_actual      = [pos[1] for pos in controller.mearsure_path]
    angle_actual  = [pos[2] for pos in controller.mearsure_path]
    x_desired     = [pos[0] for pos in controller.desired_path]
    y_desired     = [pos[1] for pos in controller.desired_path]
    angle_desired = [pos[2] for pos in controller.desired_path]
    plt.figure(figsize=(8, 6))

    # # Plot x
    # plt.subplot(3, 1, 1)
    # plt.plot(time_stamps, x_actual, label="x_actual", color='blue', linewidth = 0.5)
    # plt.plot(time_stamps, x_desired, label="x_desired", color='red', linestyle='--')
    # plt.ylabel("X Position (m)")
    # plt.title("X Position vs X Reference")
    # plt.legend()
    # plt.grid(True)
    # plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))  # <- Chia mỗi 0.5 m

    # # Plot y
    # plt.subplot(3, 1, 2)
    # plt.plot(time_stamps, y_actual, label="y_actual", color='green', linewidth = 0.5)
    # plt.plot(time_stamps, y_desired, label="y_desired", color='orange', linestyle='--')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Y Position (m)")
    # plt.title("Y Position vs Y Reference")
    # plt.legend()
    # plt.grid(True)
    # plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))  # <- Chia mỗi 0.5 m

    #  # Plot angle
    # plt.subplot(3, 1, 3)
    # plt.plot(time_stamps, angle_actual, label="angle_actual", color='purple', linewidth = 0.5)
    # plt.plot(time_stamps, angle_desired, label="angle_desired", color='brown', linestyle='--')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angle (rad)")
    # plt.title("Angle vs Angle Reference")
    # plt.legend()
    # plt.grid(True)
    # plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))  # <- Chia mỗi 0.5 rad
    # plt.tight_layout()
    # plt.show()
    # --- Plot X ---
    plt.figure(figsize=(8, 4))
    plt.plot(time_stamps, x_actual, label="x_actual", color='blue', linewidth=0.5)
    plt.plot(time_stamps, x_desired, label="x_desired", color='red', linestyle='--')
    plt.ylabel("X Position (m)")
    plt.title("X Position vs X Reference")
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))  # 0.2 m mỗi vạch chia
    plt.xlabel("Time (s)")
    plt.tight_layout()

    # --- Plot Y ---
    plt.figure(figsize=(8, 4))
    plt.plot(time_stamps, y_actual, label="y_actual", color='green', linewidth=0.5)
    plt.plot(time_stamps, y_desired, label="y_desired", color='orange', linestyle='--')
    plt.ylabel("Y Position (m)")
    plt.title("Y Position vs Y Reference")
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
    plt.xlabel("Time (s)")
    plt.tight_layout()

    # --- Plot Angle ---
    plt.figure(figsize=(8, 4))
    plt.plot(time_stamps, angle_actual, label="angle_actual", color='purple', linewidth=0.5)
    plt.plot(time_stamps, angle_desired, label="angle_desired", color='brown', linestyle='--')
    plt.ylabel("Angle (rad)")
    plt.title("Angle vs Angle Reference")
    plt.legend()
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.02))  # 0.01 rad mỗi vạch chia
    plt.xlabel("Time (s)")
    plt.tight_layout()

    plt.show()

def plot_velocity(controller):
    time_stamps = controller.error_time

    # Velocity error
    x_dot_traj = controller.desired_vx
    y_dot_traj = controller.desired_vy
    actual_vx  = controller.actual_vx
    actual_vy  = controller.actual_vy
    plt.figure(figsize=(10, 8))

    # --- Vx ---
    plt.subplot(2, 1, 1)
    plt.plot(time_stamps, actual_vx, label="actual_vx", color='blue', linewidth=0.5)
    plt.plot(time_stamps, x_dot_traj, label="x_dot_traj", color='red', linestyle='--')
    plt.ylabel("vx (m/s)")
    plt.title("Velocity Tracking - vx")
    plt.legend()
    plt.grid(True)

    # --- Vy ---
    plt.subplot(2, 1, 2)
    plt.plot(time_stamps, actual_vy, label="actual_vy", color='blue', linewidth=0.5)
    plt.plot(time_stamps, y_dot_traj, label="y_dot_traj", color='red', linestyle='--')
    plt.ylabel("vy (m/s)")
    plt.title("Velocity Tracking - vy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_velocity_error(controller):
    time_stamps = controller.error_time

    # Velocity error
    error_vx   = controller.error_velo_x  # u_history lưu [vx, vy, wz]
    error_vy   = controller.error_velo_y

    plt.figure(figsize=(10, 8))

    # --- Vx ---
    plt.subplot(2, 1, 1)
    plt.plot(time_stamps, error_vx, 'r-', linewidth=0.5, label="error_velo_x")
    plt.ylabel("vx (m/s)")
    plt.title("Velocity Tracking Error - vx")
    plt.legend()
    plt.grid(True)

    # --- Vy ---
    plt.subplot(2, 1, 2)
    plt.plot(time_stamps, error_vy, 'b-', linewidth=0.5, label="error_velo_y")
    plt.ylabel("vy (m/s)")
    plt.title("Velocity Tracking Error - vy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_control_inputs(controller):
    u_history = np.array(controller.u_history)
    time_stamps = controller.error_time

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time_stamps, u_history[:, 0], label="vx - linear.x", color='blue', linewidth = 0.5)
    plt.ylabel("vx (m/s²)")
    plt.title("Control Input vx")
    plt.grid(False)

    plt.subplot(3, 1, 2)
    plt.plot(time_stamps, u_history[:, 1], label="vy - linear.y", color='green', linewidth = 0.5)
    plt.ylabel("vy (m/s²)")
    plt.title("Control Input vy")
    plt.grid(False)

    plt.subplot(3, 1, 3)
    plt.plot(time_stamps, u_history[:, 2], label="omega - angular.z", color='red', linewidth = 0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("omega (rad/s²)")
    plt.title("Control Input u[2]")
    plt.grid(False)

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    try:   
        controller = BacksteppingController()
    except rospy.ROSInterruptException:
        pass
    finally:
        plot_paths(controller.desired_path, controller.actual_path, controller.mearsure_path)
        plot_position_vs_reference(controller) 
        plot_velocity(controller)
        plot_velocity_error(controller)
        plot_control_inputs(controller)

