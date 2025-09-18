#!/usr/bin/env python3

import numpy as np
import math
import cv2
import cv2.aruco as aruco
import socket
import time
from gui_ui import Ui_MainWindow

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
            # x_smooth, y_smooth, phi_smooth = smooth_pose(x_filt, y_filt, phi_filt)
            x_kf = kf_x.update(x_filt)
            y_kf = kf_y.update(y_filt)
            phi_kf = kf_phi.update(phi_filt)

            pose = {
                'x': x_kf,
                'y': y_kf,
                'phi': phi_kf * math.pi / 180
            }

            send_pose(pose['x'], pose['y'], pose['phi'])
            time.sleep(0.01)

            cv2.putText(frame, f"X={pose['x']:.3f}, Y={pose['y']:.3f}, Phi={pose['phi']:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
