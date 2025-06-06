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

def generate_capsule_path(radius, height, points_per_arc=50, points_per_line=50):
    # Góc cho nửa tròn
    theta_top = np.linspace(np.pi, 0, points_per_arc)        # từ trái sang phải (nửa trên)
    theta_bottom = np.linspace(0, -np.pi, points_per_arc)    # từ phải sang trái (nửa dưới)

    # Nửa tròn trên (trái -> phải)
    top_arc_x = radius * np.cos(theta_top)
    top_arc_y = radius * np.sin(theta_top) + height / 2

    # Đoạn thẳng bên phải (trên -> dưới)
    # right_line_x = np.linspace(radius, radius, points_per_line)
    # right_line_y = np.linspace(height / 2, -height / 2, points_per_line)

    # # Nửa tròn dưới (phải -> trái)
    # bottom_arc_x = radius * np.cos(theta_bottom)
    # bottom_arc_y = radius * np.sin(theta_bottom) - height / 2

    # # Đoạn thẳng bên trái (dưới -> trên)
    # left_line_x = np.linspace(-radius, -radius, points_per_line)
    # left_line_y = np.linspace(-height / 2, height / 2, points_per_line)

    # Ghép các đoạn lại theo thứ tự liên tục
    x_traject  = np.concatenate([top_arc_x])
    y_traject  = np.concatenate([top_arc_y])
    
    # x_traject  = np.concatenate([left_line_x])
    # y_traject  = np.concatenate([left_line_y])
        # Tính góc theo trục X cho mỗi điểm (trừ điểm đầu và cuối)
    angles_traject = []
    for i in range(1, len(x_traject) - 1):
        dx = x_traject[i + 1] - x_traject[i - 1]
        dy = y_traject[i + 1] - y_traject[i - 1]
        angle = np.arctan2(dy, dx)  # Tính góc theo trục X
        angles_traject.append(angle)

    # Thêm góc cho điểm đầu và cuối (lấy từ điểm tiếp theo và trước đó)
    angles_traject = [angles_traject[0]] + angles_traject + [angles_traject[-1]]
    return x_traject, y_traject, angles_traject

def generate_circle_path(radius, points):
# Điểm bắt đầu tại góc π (x = -1, y = 0), đi theo chiều kim đồng hồ → giảm góc
    theta = np.linspace(-np.pi/2, -np.pi/2 - 2.5*np.pi, points, endpoint=True)

    # Tọa độ các điểm trên đường tròn
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # # Tính hướng chuyển động (tiếp tuyến theo chiều kim đồng hồ)
    # angles = []
    # for i in range(1, len(x) - 1):
    #     dx = x[i + 1] - x[i - 1]
    #     dy = y[i + 1] - y[i - 1]
    #     angle = np.arctan2(dy, dx)
    #     angle = (angle + np.pi) % (2 * np.pi) - np.pi

    #     angles.append(angle)
    # # Thêm hướng cho điểm đầu và cuối
    # angles = [angles[0]] + angles + [angles[-1]]
    # Tính hướng (tiếp tuyến) của quỹ đạo
    angles = []
    for i in range(1, len(x) - 1):
        dx = x[i + 1] - x[i - 1]
        dy = y[i + 1] - y[i - 1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)

    # Dùng unwrap để được góc liên tục tăng/giảm đều (không nhảy ±2pi)
    angles = np.unwrap(angles)

    # Dịch toàn bộ góc sao cho điểm bắt đầu là pi
    angle_shift = np.pi - angles[0]
    angles = angles + angle_shift

    # Wrap lại về [-pi, pi] nếu cần
    angles = np.array([wrap_to_pi(a) for a in angles])

    # Thêm hướng cho điểm đầu và cuối
    angles = np.concatenate(([angles[0]], angles, [angles[-1]]))
    return x, y, angles
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

    def keyboard_listener(self):
        keyboard.wait('q')
        rospy.loginfo("Stopping robot due to 'q' key press.")
        self.running = False
        for i in range(0,3):    
            self.stop_robot()
        rospy.signal_shutdown("Stopped by user.")

    def odom_callback(self, msg):
        """ Update robot's state from Float32MultiArray. """
        # Lấy giá trị từ message Float32MultiArray
        if (self.trajectory_index < self.N):
            self.x = msg.data[0]  # pose['x']
            self.y = msg.data[1]  # pose['y']
            self.theta = msg.data[2]  # pose['phi']
            self.v = msg.data[3]
            self.pose_x = msg.data[4]
            self.pose_y = msg.data[5]
            self.pose_phi = msg.data[6]
            # self.x_d = 1  # pose['x']
            # self.y_d = 1  # pose['y']
            # self.theta_d = 0.1  # pose['phi']
            # self.v = 0.1
            
            self.x_dot = self.v * math.cos(self.theta)
            self.y_dot = self.v * math.sin(self.theta) 
        else:
            self.x = 0
            self.y = 0
            self.theta = 0
            self.v = 0
            self.pose_x = 0
            self.pose_y = 0
            self.pose_phi = 0
            self.x_dot = 0
            self.y_dot = 0 
        #print(f"y_dot: {self.y_dot}")
    
    def uart_callback(self, msg):
        if (self.trajectory_index < self.N):
            self.theta_dot = msg.data[2]
        else:
            self.theta_dot = 0

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
        self.k1 = 15
        self.k2 = 0.8
        self.plot_initialized = False

        # Robot state
        self.x = 0.0 
        self.y = -1.0
        self.theta = math.pi
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.theta_dot = 0.0
        self.v = 0.0
        self.pose_x = 0.0
        self.pose_y = -1.0
        self.pose_phi = math.pi
        # Thông số quỹ đạoq
        # self.T = 32.0       # Tổng thời gian (s)
        # self.T = 40.0       # Tổng thời gian (s)
        self.T = 125.0       # Tổng thời gian (s)

        self.dt = 0.01     # Bước thời gian (s)
        self.N = int(self.T / self.dt)
        self.trajectory_index = 0
        
        # self.x_traject, self.y_traject, self.angles_traject = generate_capsule_path(radius=1, height=2, points_per_arc=self.N, points_per_line= self.N)
        self.x_traject, self.y_traject, self.angles_traject = generate_circle_path(1 ,self.N)

        self.path_points = list(zip(self.x_traject, self.y_traject, self.angles_traject))

        # Ve quy dao 
        self.error_x = []
        self.error_y = []
        self.error_time = []
        self.actual_path = []
        self.desired_path = [] 
        self.mearsure_path = []
        self.u_history = []
        # Stop variable
        self.running = True
        threading.Thread(target=self.keyboard_listener, daemon=True).start()
        # Tạo quỹ đạo tham chiếu (linear interpolation)
        # self.x_traj = np.linspace( -0.35,  -0.35, self.N)  # x không đổi
        # self.y_traj = np.linspace(-1.4, 0, self.N) # y từ -1 lên 0
        # self.theta_traj = np.linspace(np.pi/2, np.pi/2, self.N)  # không đổi hướng
        self.x_traj = [pt[0] for pt in self.path_points]
        self.y_traj = [pt[1] for pt in self.path_points]
        self.theta_traj = [pt[2] for pt in self.path_points]

        # Gọi timer để cập nhật trajectory mỗi dt giây
        rospy.Timer(rospy.Duration(self.dt), self.update_trajectory)

        rospy.Timer(rospy.Duration(self.dt), self.control_loop)
        rospy.spin()
    
    def update_trajectory(self, event):
        if self.trajectory_index < self.N-1:
            print(f"idex: {self.trajectory_index} and N {self.N}")
            self.x_d = self.x_traj[self.trajectory_index]
            self.y_d = self.y_traj[self.trajectory_index]
            self.theta_d = self.theta_traj[self.trajectory_index]
            self.x_dot_d = (self.x_traj[self.trajectory_index + 1] - self.x_traj[self.trajectory_index]) / self.dt
            self.y_dot_d = (self.y_traj[self.trajectory_index + 1] - self.y_traj[self.trajectory_index]) / self.dt
            self.theta_dot_d = (self.theta_traj[self.trajectory_index + 1] - self.theta_traj[self.trajectory_index]) / self.dt
            self.trajectory_index += 1
        else:
            # Giữ trạng thái cuối cùng nếu đã vượt quá quỹ đạo
            self.x = self.x_d
            self.y = self.y_d
            self.theta = self.theta_d
            # self.x_d = 0
            # self.y_d = 0
            # self.theta_d = 0
            # self.x_dot_d = 0.0
            # self.y_dot_d = 0.0
            # self.theta_dot_d = 0.0

            # # Cập nhật cờ để dừng điều khiển
            # self.running = False  # <- Dừng vòng lặp điều khiển (control_loop)

            # # Gửi tín hiệu dừng liên tục
            # stop_cmd = Twist()
            # stop_cmd.linear.x = 0.0
            # stop_cmd.linear.y = 0.0
            # stop_cmd.angular.z = 0.0
            # self.cmd_pub.publish(stop_cmd)

    def control_loop(self, event):
        # if not self.running:
        #     # # Gửi lệnh dừng liên tục
        #     # cmd = Twist()
        #     # cmd.linear.x = 0.0
        #     # cmd.linear.y = 0.0
        #     # cmd.angular.z = 0.0
        #     # self.cmd_pub.publish(cmd)
        #     return
        """ Apply backstepping control. """

        e1 = np.array([self.x_d - self.x, self.y_d - self.y, self.theta_d - self.theta])
        e1[2] = wrap_to_pi(e1[2])  # Gói góc sai số về [-pi, pi]
        e2 = np.array([self.x_dot_d - self.x_dot, self.y_dot_d - self.y_dot, self.theta_dot_d - self.theta_dot])

        # Velocity predict
        eta_dot_d = np.array([self.x_dot_d, self.y_dot_d, self.theta_dot_d])
        v_d = eta_dot_d + self.k1 * e1

        # Áp dụng ma trận nghịch đảo Jacobian
        J = np.array([[np.cos(self.theta), -np.sin(self.theta),     0],
                      [np.sin(self.theta),  np.cos(self.theta),     0],
                      [0,                   0,                      1]])
        J_inv = np.linalg.pinv(J)  # Dùng pseudo-inverse thay vì nghịch đảo
        
        u = np.dot(D(), (self.k2 * e2 + np.dot(J_inv, v_d) + np.dot(J.T, e1))) + self.n_zeta(self.x_dot, self.y_dot, self.theta_dot)

        # Publish command
        cmd = Twist()
        # u[1], u[0], u[2] = LowPassFilter(u[1], u[0], u[2])

        cmd.linear.x = u[1]
        cmd.linear.y = u[0]
        cmd.angular.z = u[2]
        
        self.cmd_pub.publish(cmd)

        # Draw path
        self.u_history.append([u[1], u[0], u[2]])
        self.mearsure_path.append((self.x, self.y, self.theta))
        self.actual_path.append((self.pose_x, self.pose_y, self.pose_phi))
        self.desired_path.append((self.x_d, self.y_d, self.theta_d))

        self.error_x.append(abs(self.x_d - self.pose_x))
        self.error_y.append(abs(self.y_d - self.pose_y))
        
        self.error_time.append(self.trajectory_index * self.dt)  # hoặc rospy.get_time() nếu cần chính xác tuyệt đối
        print(f"x_a = {self.pose_x}, y_a = {self.pose_y}")
        print(f"x_m = {self.x}, y_m = {self.y}, a_m = {self.theta}")
        print(f"x_d = {self.x_d}, y_d = {self.y_d}, a_d = {self.theta_d}")

def plot_paths(desired, actual, mearsure):
    desired  = np.array(desired)
    actual   = np.array(actual)
    mearsure = np.array(mearsure)

    plt.figure(figsize = (8, 6))
    plt.plot(desired[:, 0], desired[:, 1], 'r-', linewidth = 0.5, label='Desired Path')
    plt.plot(mearsure[:, 0], mearsure[:, 1], 'b-', linewidth = 1, label='Mearsure - Aruko Path')
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
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))  # 0.5 rad mỗi vạch chia
    plt.xlabel("Time (s)")
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
        plot_control_inputs(controller)

