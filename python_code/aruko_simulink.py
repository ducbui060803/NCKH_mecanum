#!/usr/bin/env python3

import numpy as np
import math
import cv2
import cv2.aruco as aruco
import socket
import time
from gui_ui import Ui_MainWindow
import matplotlib.pyplot as plt

# --- Plot yaw compare ---
x_kalman_list  = []
x_normal_list  = []
y_kalman_list  = []
y_normal_list  = []
yaw_kalman_list  = []
yaw_normal_list  = []

x_kalman_error = []
x_normal_error = []
y_kalman_error = []
y_normal_error = []
yaw_kalman_error = []
yaw_normal_error = []

x_kalman_error_mean = 0
x_normal_error_mean = 0
y_kalman_error_mean = 0
y_normal_error_mean = 0
yaw_kalman_error_mean = 0
yaw_normal_error_mean = 0
x_kalman_error_max = 0
x_normal_error_max = 0
y_kalman_error_max = 0
y_normal_error_max = 0
yaw_kalman_error_max = 0
yaw_normal_error_max = 0

yaw_desired_list = []
x_desired_list = []
y_desired_list = []
time_stamps = []
start_time = time.time()
x_desired = 0 # Simulink desired x
y_desired = 0 # Simulink desired y
yaw_desired = 90 # Simulink desired yaw
def plot_yaw_kalman():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, yaw_normal_list, 'r-', label="yaw_normal", linewidth=0.8)
    plt.plot(time_stamps, yaw_kalman_list, 'b-', label="yaw_kalman ", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("yaw (degree)")
    plt.title("Comparison of yaw_kalman vs yaw_normal")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# def plot_yaw_normal():
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_stamps, yaw_normal_list, 'b-', label="yaw_normal ", linewidth=0.8)
#     plt.plot(time_stamps, yaw_desired_list, 'r--', label="yaw_desired", linewidth=0.8)
#     plt.xlabel("Time (s)")
#     plt.ylabel("yaw (degree)")
#     plt.title("Comparison of yaw_normal vs yaw_desired")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
def plot_x_kalman():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, x_normal_list, 'r-', label="x_normal", linewidth=0.8)
    plt.plot(time_stamps, x_kalman_list, 'b-', label="x_kalman ", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("x (m)")
    plt.title("Comparison of x_kalman vs x_normal")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# def plot_x_normal():
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_stamps, x_normal_list, 'g-', label="x_normal ", linewidth=0.8)
#     plt.plot(time_stamps, x_desired_list, 'r--', label="x_desired", linewidth=0.8)
#     plt.xlabel("Time (s)")
#     plt.ylabel("x(m)")
#     plt.title("Comparison of x_normal vs x_desired")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
def plot_y_kalman():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, y_normal_list, 'r-', label="y_normal", linewidth=0.8)
    plt.plot(time_stamps, y_kalman_list, 'b-', label="y_kalman ", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("y (m)")
    plt.title("Comparison of y_kalman vs y_normal")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# def plot_y_normal():
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_stamps, y_normal_list, '-', color='purple', label="y_normal ", linewidth=0.8)
#     plt.plot(time_stamps, y_desired_list, 'r--', label="y_desired", linewidth=0.8)
#     plt.xlabel("Time (s)")
#     plt.ylabel("y (m)")
#     plt.title("Comparison of y_normal vs y_desired")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# Simulink Global variable
v = 0
pre_v = 0
acc_prev = 0
yaw_dot = 0
imu_yaw = 0
# Khởi tạo EKF với ma trận Q và R cố định
dt = 0.01
Q_fixed = [0.2, 0.2, 0.1, 0.02]
R_fixed = [0.005, 0.005, 0.001]  # Ma trận nhiễu đo lường cố định


# --- Params giả lập ---
encoder_noise_std = 0.02     # m/s (std dev của encoder)
imu_noise_std = 0.005        # rad/s (std dev của gyro)
imu_bias_walk_std = 0.0005   # bias random walk per step (rad/s)
camera_dropout_prob = 0.03   # xác suất frame camera bị "mất" (3%)
simulate_latency = 0.0       # giây, nếu muốn thử latency, set > 0

# trạng thái nhớ để tính đạo hàm
_sim_prev_time = None
_sim_prev_pose = None  # (x, y, phi)
_imu_bias = 0.0

import random
def _wrap_angle_rad(a):
    return math.atan2(math.sin(a), math.cos(a))

def simulate_sensors_from_pose(curr_pose, t_now):
    """
    curr_pose: (x, y, phi) in meters and radians (phi in radians)
    t_now: timestamp in seconds (time.time())
    returns: v_enc, yaw_rate_imu, camera_available (bool)
    """
    global _sim_prev_time, _sim_prev_pose, _imu_bias

    # Simulate random camera dropout
    if random.random() < camera_dropout_prob:
        camera_available = False
    else:
        camera_available = True

    # If no previous pose, return zeros (no motion yet)
    if _sim_prev_pose is None or _sim_prev_time is None:
        _sim_prev_pose = curr_pose
        _sim_prev_time = t_now
        return 0.0, 0.0, camera_available

    dt_sim = max(1e-6, t_now - _sim_prev_time)

    # Finite difference for linear speed (approx)
    dx = curr_pose[0] - _sim_prev_pose[0]
    dy = curr_pose[1] - _sim_prev_pose[1]
    v_true = math.sqrt(dx*dx + dy*dy) / dt_sim

    # Finite difference for yaw rate (unwrap)
    dphi = _wrap_angle_rad(curr_pose[2] - _sim_prev_pose[2])
    yaw_rate_true = dphi / dt_sim

    # add encoder noise
    v_enc = v_true + np.random.normal(0.0, encoder_noise_std)

    # IMU: add bias + noise; and evolve bias as a random walk
    _imu_bias += np.random.normal(0.0, imu_bias_walk_std)
    yaw_imu = yaw_rate_true + _imu_bias + np.random.normal(0.0, imu_noise_std)

    # update prev
    _sim_prev_pose = curr_pose
    _sim_prev_time = t_now

    return v_enc, yaw_imu, camera_available

# --- Extended Kalman Filter ---
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
        yaw_t = self.x_t[2, 0]  # Góc quay hiện tại
        v_t = self.x_t[3, 0]    # Vận tốc hiện tại

        # Ma trận Jacobian F_t
        F_t = np.array([
            [1, 0, -self.dt * v_t * np.sin(yaw_t), self.dt * np.cos(yaw_t)],
            [0, 1, self.dt * v_t * np.cos(yaw_t), self.dt * np.sin(yaw_t)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        acc = 0.8 * acc_prev + 0.2 * (u_t[0] - pre_v)

        # Ma trận điều khiển B_t
        B_t = np.array([
                [np.cos(yaw_t) * dt * u_t[0]],
                [np.sin(yaw_t) * dt * u_t[0]],
                [u_t[1]],
                # [u_t[0]-pre_v]
                [acc]
            ])
        
        # Dự đoán trạng thái mới
        self.x_t = self.x_t + B_t  # x_t|t-1 = f(x_t-1, u_t)


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
    
ekf = EKF(dt, Q_fixed, R_fixed)
#---------------------------------------------------------------------------------------------------------#
# --- Hàm lọc ---
updated_id11_once = False
updated_id4_once = False

def dis_filter(current_value, pre_ar, a):
    if a == 1:
        thr = 3
    else:
        thr = 0.03
        
    if pre_ar == 0:
        pre_ar = current_value
        return True, pre_ar

    if abs(current_value - pre_ar) < thr:
        pre_ar = current_value
        return True, pre_ar

    return False, pre_ar

# --- Socket ---
IPC_IP = "172.18.223.255"   # Thay bằng IP của IPC
PORT = 5005  

def send_pose(x, y, phi):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        pose_str = f"{x},{y},{phi}"
        sock.sendto(pose_str.encode(), (IPC_IP, PORT))
        print(f"Sent: {pose_str}")

# --- Camera calibration ---
camera_calibration_parameters_filename = 'calibration_chessboard_webcam.yaml'
cv_file = cv2.FileStorage(camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode('K').mat()
dist_coeffs = cv_file.getNode('D').mat()

# --- Tính toán pose ---
def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [ marker_size / 2, marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)
    
    trash, rvecs, tvecs = [], [], []
    for c in corners:
        _, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(_)
    return rvecs, tvecs, trash

def get_aruco(x0, y0, phi0, rvect, tvect):
    phi1 = math.atan(-rvect[2][0] / math.sqrt(rvect[2][1]**2 + rvect[2][2]**2))
    d = math.sqrt(tvect[0]**2 + (tvect[2] + 0.11)**2)
    phiaruco = phi1 + phi0
    phi2 = math.atan(tvect[0] / (tvect[2] + 0.11))
    phi3 = phiaruco - phi2
    xaruco = x0 - d * math.cos(phi3)
    yaruco = y0 - d * math.sin(phi3)
    return xaruco, yaruco, phiaruco, d

def find_min_index(numbers):
    return min(range(len(numbers)), key=lambda i: numbers[i])

def marker_area(corner):
    pts = corner[0]
    return cv2.contourArea(pts.astype(np.float32))

def normalize_angle_deg(angle):
    angle = (angle + 180) % 360 - 180
    return 180 if angle == -180 else angle

# --- Khởi tạo camera ---
cap = cv2.VideoCapture(0)  # Có thể thay bằng URL camera IP
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# --- ArUco ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
marker_size = 0.145 

# --- Marker map ---
id_12 = [-1,  -1,  -math.pi]
id_11 = [-0.4, -1.4, -math.pi]
id_10 = [0.4, -1.4, -math.pi/2 ]
id_9  = [1,   -1,   -math.pi/2 ]
id_8  = [1.4, -0.4, -math.pi/2]
id_7  = [1.4,  0.4,  0]
id_6  = [1,    1,    0]
id_5  = [0.4,  1.4,  0]
id_4  = [-0.4, 1.4,  math.pi/2]
id_3  = [-1,   1,    math.pi/2]
id_2  = [0,    0,    math.pi/2]
id_1  = [1, 2,  math.pi/2]
id_0  = [0,  1,    math.pi/2]

marker_start = [id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_7, id_8, id_9, id_10, id_11, id_12]

# --- Biến toàn cục ---
pre_x, pre_y, pre_angle = 0, -1, 180
pose_buffer = []
alpha, a_a = 0.5, 0.5
filtered_x = filtered_y = filtered_angle = None

class SimpleKalman:
    def __init__(self, q=0.01, r=0.1):
        self.q = q  # process noise
        self.r = r  # measurement noise
        self.p = 1.0
        self.x = 0.0

    def update(self, measurement):
        # predict
        self.p += self.q
        # kalman gain
        k = self.p / (self.p + self.r)
        # update estimate
        self.x += k * (measurement - self.x)
        # update error covariance
        self.p *= (1 - k)
        return self.x

# --- tạo bộ lọc cho x,y,phi
kf_x = SimpleKalman()
kf_y = SimpleKalman()
kf_phi = SimpleKalman()

def LowPassFilter(x, y, angle):
    global filtered_x, filtered_y, filtered_angle, alpha, a_a
    if filtered_x is None or filtered_y is None or filtered_angle is None:
        filtered_x, filtered_y, filtered_angle = x, y, angle
    else:
        filtered_x = alpha * x + (1 - alpha) * filtered_x
        filtered_y = alpha * y + (1 - alpha) * filtered_y
        filtered_angle = a_a * angle + (1 - a_a) * filtered_angle
    return filtered_x, filtered_y, filtered_angle

def should_filter_angle(vx, vy):
    return math.sqrt(vx**2 + vy**2) > 0.1  

def smooth_pose(x, y, phi, window=5):
    global pose_buffer
    pose_buffer.append([x, y, phi])
    if len(pose_buffer) > window:
        pose_buffer.pop(0)
    arr = np.array(pose_buffer)
    smoothed = np.mean(arr, axis=0)
    
    if len(pose_buffer) >= 2:
        delta = np.linalg.norm(arr[-1, :2] - arr[-2, :2])
        if delta < 0.015:
            smoothed[:2] = arr[-2, :2]
    return smoothed[0], smoothed[1], smoothed[2]

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        all_marker, distance_infor = [], []
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size)

            rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
            id = ids[i][0]
            if id <= 13:
                x0, y0, phi0 = marker_start[id]
                xaruco, yaruco, phiaruco, distance = get_aruco(x0, y0, phi0, rotation_matrix, tvecs[i])
                phiaruco = normalize_angle_deg(math.degrees(phiaruco))

                marker_info = [xaruco, yaruco, phiaruco]
                all_marker.append(marker_info)
                distance_infor.append(distance)

        if all_marker:
            weights = 1 / (np.array(distance_infor) + 1e-6)
            weights /= np.sum(weights)
            x_est = np.sum(weights * np.array([p[0] for p in all_marker]))
            y_est = np.sum(weights * np.array([p[1] for p in all_marker]))
            phi_est = np.sum(weights * np.array([p[2] for p in all_marker]))

            x_filt, y_filt, phi_filt = LowPassFilter(x_est, y_est, phi_est)
            x_smooth, y_smooth, phi_smooth = smooth_pose(x_filt, y_filt, phi_filt)

            x_kf = kf_x.update(x_smooth)
            y_kf = kf_y.update(y_smooth)
            phi_kf = kf_phi.update(phi_smooth)

            pose = {
                'x': x_kf,
                'y': y_kf,
                'phi': phi_kf * math.pi / 180
            }

            # EKF Check
            v = math.sqrt(0.1*0.1 + 0) # simulink vx = 0.1 m/s vy = 0 
            yaw_dot = 0  # simulink yaw rate = 0 rad/s
            u_t = np.array([v , yaw_dot])
            z_t = np.array([[x_kf], [y_kf], [phi_kf]])
            # Bước dự đoán và cập nhật
            ekf.predict(u_t)
            ekf.update(z_t)
            state = ekf.get_state().flatten()
            pre_v = v
            
            send_pose(pose['x'], pose['y'], pose['phi'])
            time.sleep(0.01)

            cv2.putText(frame, f"X_aruco={pose['x']:.3f}, Y_aruco={pose['y']:.3f}, Phi_aruco={pose['phi']:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Lưu giá trị để plot
            now = time.time() - start_time
            time_stamps.append(now)
            yaw_kalman_list.append(phi_kf) 
            yaw_normal_list.append(phi_est) 
            x_kalman_list.append(x_kf) 
            x_normal_list.append(x_est) 
            y_kalman_list.append(y_kf) 
            y_normal_list.append(y_est)
            
            yaw_kalman_error.append(abs(phi_kf - yaw_desired))
            yaw_normal_error.append(abs(phi_est - yaw_desired))
            x_kalman_error.append(abs(x_kf - x_desired))
            x_normal_error.append(abs(x_est - x_desired))
            y_kalman_error.append(abs(y_kf - y_desired))
            y_normal_error.append(abs(y_est - y_desired))
            
            yaw_desired_list.append(yaw_desired)
            x_desired_list.append(x_desired)
            y_desired_list.append(y_desired)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



plot_x_kalman()
# plot_x_normal()
plot_y_kalman()
# plot_y_normal()
plot_yaw_kalman()
# plot_yaw_normal()
yaw_kalman_error_mean = sum(yaw_kalman_error)/len(yaw_kalman_error)
yaw_normal_error_mean = sum(yaw_normal_error)/len(yaw_normal_error)
x_kalman_error_mean = sum(x_kalman_error)/len(x_kalman_error)
x_normal_error_mean = sum(x_normal_error)/len(x_normal_error)
y_kalman_error_mean = sum(y_kalman_error)/len(y_kalman_error)
y_normal_error_mean = sum(y_normal_error)/len(y_normal_error)

yaw_kalman_error_max = max(yaw_kalman_error)
yaw_normal_error_max = max(yaw_normal_error)
x_kalman_error_max = max(x_kalman_error)
x_normal_error_max = max(x_normal_error)
y_kalman_error_max = max(y_kalman_error)
y_normal_error_max = max(y_normal_error)

print('----------------------------------')
print(f'x_kalman_error_max: {x_kalman_error_max}')
print(f'x_normal_error_max: {x_normal_error_max}')
print(f'y_kalman_error_max: {y_kalman_error_max}')
print(f'y_normal_error_max: {y_normal_error_max}')
print(f'yaw_kalman_error_max: {yaw_kalman_error_max}')
print(f'yaw_normal_error_max: {yaw_normal_error_max}')
print('----------------------------------')
print(f'x_kalman_error_mean: {x_kalman_error_mean}')
print(f'x_normal_error_mean: {x_normal_error_mean}')
print(f'y_kalman_error_mean: {y_kalman_error_mean}')
print(f'y_normal_error_mean: {y_normal_error_mean}')
print(f'yaw_kalman_error_mean: {yaw_kalman_error_mean}')
print(f'yaw_normal_error_mean: {yaw_normal_error_mean}')
print('----------------------------------')

cap.release()
cv2.destroyAllWindows()
