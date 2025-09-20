#!/usr/bin/env python3
import sys, math, time, cv2, cv2.aruco as aruco
import numpy as np
import socket
import json
from PyQt5 import QtWidgets, QtGui, QtCore
from gui_ui import Ui_MainWindow
# ---------------------------- Biến toàn cục ----------------------------
pose_buffer = []
alpha, a_a = 0.5, 0.5
filtered_x = filtered_y = filtered_angle = None
# ---------------------------- Marker map ----------------------------
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
#---------------------------- End Marker map ----------------------------
# ---------------------------- Socket ----------------------------
IPC_IP = "172.18.223.255"   # Thay bằng IP của IPC
PORT = 5005  

# # ---------------------------- Hàm gửi pose qua UDP ----------------------------
# def send_pose(x, y, yaw):
#     with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
#         pose_str = f"{x},{y},{yaw}"
#         sock.sendto(pose_str.encode(), (IPC_IP, PORT))
#         print(f"Sent: {pose_str}")
        
# ---------------------------- Hàm lọc ----------------------------

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

# ---------------------------- Hàm chuẩn hóa góc về [-180:180] ----------------------------
def normalize_angle_deg(angle):
    angle = (angle + 180) % 360 - 180
    return 180 if angle == -180 else angle

# ---------------------------- Bộ lọc Kalman đơn giản ----------------------------
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

# ---------------------------- Hàm lọc thông thấp ----------------------------
def LowPassFilter(x, y, angle):
    global filtered_x, filtered_y, filtered_angle, alpha, a_a
    if filtered_x is None or filtered_y is None or filtered_angle is None:
        filtered_x, filtered_y, filtered_angle = x, y, angle
    else:
        filtered_x = alpha * x + (1 - alpha) * filtered_x
        filtered_y = alpha * y + (1 - alpha) * filtered_y
        filtered_angle = a_a * angle + (1 - a_a) * filtered_angle
    return filtered_x, filtered_y, filtered_angle

# ---------------------------- Hàm kiểm tra có nên lọc góc hay không ----------------------------
def should_filter_angle(vx, vy):
    return math.sqrt(vx**2 + vy**2) > 0.1  

# ---------------------------- Hàm làm mượt pose ----------------------------
def smooth_pose(x, y, yaw, window=5):
    global pose_buffer
    pose_buffer.append([x, y, yaw])
    if len(pose_buffer) > window:
        pose_buffer.pop(0)
    arr = np.array(pose_buffer)
    smoothed = np.mean(arr, axis=0)
    
    if len(pose_buffer) >= 2:
        delta = np.linalg.norm(arr[-1, :2] - arr[-2, :2])
        if delta < 0.015:
            smoothed[:2] = arr[-2, :2]
    return smoothed[0], smoothed[1], smoothed[2]

# ---------------------------- Camera calibration ----------------------------
camera_calibration_parameters_filename = 'calibration_chessboard_webcam.yaml'
cv_file = cv2.FileStorage(camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode('K').mat()
dist_coeffs = cv_file.getNode('D').mat()

# ---------------------------- Tính toán pose ----------------------------
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

# ------------- Tính toán pose ArUco trong hệ toạ độ toàn cục -------------------
def get_aruco(x0, y0, phi0, rvect, tvect):
    phi1 = math.atan(-rvect[2][0] / math.sqrt(rvect[2][1]**2 + rvect[2][2]**2))
    d = math.sqrt(tvect[0]**2 + (tvect[2] + 0.11)**2)
    phiaruco = phi1 + phi0
    phi2 = math.atan(tvect[0] / (tvect[2] + 0.11))
    phi3 = phiaruco - phi2
    xaruco = x0 - d * math.cos(phi3)
    yaruco = y0 - d * math.sin(phi3)
    return xaruco, yaruco, phiaruco, d

# ---------------------------- Hàm tìm chỉ số của phần tử nhỏ nhất trong mảng ----------------------------
def find_min_index(numbers):
    return min(range(len(numbers)), key=lambda i: numbers[i])

# ---------------------------- Hàm tính diện tích marker ----------------------------
def marker_area(corner):
    pts = corner[0]
    return cv2.contourArea(pts.astype(np.float32))

# ---------------------------- ArUco ----------------------------                     
marker_size = 0.145                                           # Kích thước marker (m)
marker_start = [id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_7, id_8, id_9, id_10, id_11, id_12] # Thông tin các marker

# --- tạo bộ lọc cho x,y,yaw
kf_x = SimpleKalman()
kf_y = SimpleKalman()
kf_yaw = SimpleKalman()
        
# ===================== Main Application =====================
class ArucoApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        # Initialize the parent class
        super().__init__()
        # ---------------------------- Thiết lập giao diện người dùng ----------------------------
        self.setupUi(self)
        
        # ------------------------- Biến và cờ trạng thái -------------------------
        self.running = False
        self.t = 0
        self.dt = 0.02
        
        # safe defaults (important to avoid AttributeError on first update)
        self.xd = 0.0; self.yd = 0.0; self.yaw_d = 0.0
        self.vx_d = 0.0; self.vy_d = 0.0; self.yaw_dot_d = 0.0
        self.ax_d = 0.0; self.ay_d = 0.0; self.yaw_2dot_d = 0.0

        # Biến lưu data nhận từ Aruco
        self.x_aruco = 0.0; self.y_aruco = 0.0; self.yaw_aruco = 0.0
        self.x_kf = 0.0; self.y_kf = 0.0; self.yaw_kf = 0.0
        
        # Biến lưu data nhận từ Controller
        self.x_EKF = 0.0
        self.y_EKF = 0.0
        self.yaw_EKF = 0.0
        self.v_local_encoder = [0.0, 0.0, 0.0]
        self.v_local_EKF = [0.0, 0.0, 0.0]
        
        # Biến lưu data vẽ đồ thị
        self.desired_points = []
        self.actual_points = []

        # ---------------------------- Khởi tạo camera ----------------------------
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # chỉ giữ lại 1 frame mới nhất
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("Error: Could not open video stream")
            exit()
        
        # ------------------------------ ArUco setup ------------------------------
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # Lấy dữ liệu từ thư viên aruco
        self.parameters = aruco.DetectorParameters()                        # Lấy dữ liệu từ thư viên aruco
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Timer camera (30 FPS)
        self.timer_frame = QtCore.QTimer()
        self.timer_frame.timeout.connect(self.update_frame)
        self.timer_frame.start(30)

        # Timer plot (5 Hz)
        self.timer_plot = QtCore.QTimer()
        self.timer_plot.timeout.connect(self.update_plot)
        self.timer_plot.start(200)

        # Timer send socket (20 Hz)
        self.timer_send = QtCore.QTimer()
        self.timer_send.timeout.connect(self.send_to_ipc)
        self.timer_send.start(50)
        
        # ------------------------- connect button backend -------------------------
        self.Connect_btn.clicked.connect(self.connect_socket)
        self.Disconnect_btn.clicked.connect(self.disconnect_socket)
        self.Start_btn.clicked.connect(self.start_motion)
        self.Stop_btn.clicked.connect(self.stop_motion)
        self.Clear_Graph_btn_.clicked.connect(self.clear_graph)

    def connect_socket(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(("192.168.1.100", 5000))  # đổi IP cho IPC
            self.sock.setblocking(False)

            # Timer để check incoming data
            self.timer_socket = QtCore.QTimer()
            self.timer_socket.timeout.connect(self.check_socket_data)
            self.timer_socket.start(50)

            print("Socket connected")
        except Exception as e:
            print(f"Connect error: {e}")


    def disconnect_socket(self):
        try:
            if hasattr(self, "timer_socket"):
                self.timer_socket.stop()
                self.timer_socket.deleteLater()
                del self.timer_socket

            if hasattr(self, "sock"):
                self.sock.close()
                del self.sock

            print("Socket disconnected")
        except Exception as e:
            print(f"Disconnect error: {e}")
        
    def start_motion(self):
        self.running = True
        self.t0 = time.time()   # mốc thời gian bắt đầu
        self.t = 0
        self.desired_points.clear()
        self.actual_points.clear()
        if hasattr(self, "error_data"):
            for k in self.error_data:
                self.error_data[k].clear()
    
    def stop_motion(self):
        self.running = False

    def clear_graph(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.plot_widget.draw()
        self.plot_widget_1.draw()
        self.plot_widget_2.draw()
        self.plot_widget_3.draw()

    # ---------------------------- Data from Aruco Camera thread ----------------------------
    def handle_pose(self, x, y, yaw):
        self.x_aruco, self.y_aruco, self.yaw_aruco = x, y, yaw
        

    # ---------------------------- Data from Trạectory Thread ----------------------------
    def handle_traj(self, x, y, yaw, vx, vy, vyaw, ax, ay, ayaw):
        self.xd, self.yd, self.yaw_d = x, y, yaw
        self.vx_d, self.vy_d, self.yaw_dot_d = vx, vy, vyaw
        self.ax_d, self.ay_d, self.yaw_2dot_d = ax, ay, ayaw
    
    def update_aruco_pose(self, x, y, yaw):
        """Cập nhật giá trị ArUco vào các ô text."""
        self.x_aruco_txt.setPlainText(f"{x:.2f}")   # hiển thị 2 số lẻ
        self.y_aruco_txt.setPlainText(f"{y:.2f}")
        self.yaw_aruco_txt.setPlainText(f"{yaw:.2f}")  # giả sử yaw rad, bạn có thể đổi sang degree nếu muốn
        
    # --- Quỹ đạo mong muốn theo lựa chọn ---
    def desired_trajectory(self, t):
        yaw_traj_desired = math.pi / 2  # Giữ nguyên yaw mong muốn
        mode = self.comboBox.currentText()

        if mode == "Square":
            L = 1.0
            T = 40  # chu kỳ hoàn thành quỹ đạo (giây)
            period = T
            if t >= period:  # Dừng tại góc cuối cùng
                return 0, 0, yaw_traj_desired, 0, 0, 0, 0, 0, 0
            s = (t % period) / period
            v = 4 * L / period
            a = 0  # quỹ đạo bậc thang => gia tốc gần 0, nhưng có gián đoạn ở góc

            if s < 0.25:   # cạnh dưới
                return 4*L*s, 0, yaw_traj_desired, v, 0, 0, a, 0, 0
            elif s < 0.5:  # cạnh phải
                return L, 4*L*(s-0.25), yaw_traj_desired, 0, v, 0, 0, a, 0
            elif s < 0.75: # cạnh trên
                return L*(1 - 4*(s-0.5)), L, yaw_traj_desired, -v, 0, 0, 0, a, 0
            else:          # cạnh trái
                return 0, L*(1 - 4*(s-0.75)), yaw_traj_desired, 0, -v, 0, a, 0, 0

        elif mode == "Circle":
            R = 1.0
            T = 40  # chu kỳ hoàn thành quỹ đạo (giây)
            w = 2*math.pi/T  # tần số góc
            if t >= T:  # Dừng tại điểm cuối (x=R, y=0)
                return R, 0, yaw_traj_desired, 0, 0, 0, 0, 0, 0
            x = R*math.cos(w*t)
            y = R*math.sin(w*t)
            vx = -R*w*math.sin(w*t)
            vy =  R*w*math.cos(w*t)
            vyaw = 0
            ax = -R*(w**2)*math.cos(w*t)
            ay = -R*(w**2)*math.sin(w*t)
            ayaw = 0
            return x, y, yaw_traj_desired, vx, vy, vyaw, ax, ay, ayaw

        elif mode == "Line X":
            T = 10  # chu kỳ hoàn thành quỹ đạo (giây)
            v = 1.0/T  # tốc độ mong muốn, có thể chỉnh
            if t < T:
                return v*t, 0, yaw_traj_desired, v, 0, 0, 0, 0, 0
            else:
                return 1.0, 0, yaw_traj_desired, 0, 0, 0, 0, 0, 0

        elif mode == "Line Y":
            T = 10  # chu kỳ hoàn thành quỹ đạo (giây)
            v = 1.0/T
            if t < T:
                return 0, v*t, yaw_traj_desired, 0, v, 0, 0, 0, 0
            else:
                return 0, 1, yaw_traj_desired, 0, 0, 0, 0, 0, 0

        else:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0
        
    # --- Pose thực tế từ ArUco ---
    def get_actual_pose(self):
        return self.x_kf, self.y_kf, self.yaw_kf * math.pi / 180

    def update_plot(self):
        if not self.running:
            return

        # --- Cập nhật thời gian thực ---
        self.t = time.time() - self.t0
        
        # Quỹ đạo mong muốn
        xd, yd, yaw_d, \
        vx_global_d, vy_global_d, yaw_dot_d, \
        ax_global_d, ay_global_d, yaw_2dot_d = self.desired_trajectory(self.t) # return 9 params 
        
        # --- Dữ liệu từ Aruco ---
        x_aruco, y_aruco, yaw_aruco = self.get_actual_pose()
        self.update_aruco_pose(x_aruco, y_aruco, yaw_aruco)   
        # --- Lưu dữ liệu ---
        self.desired_points.append([xd, yd])
        self.actual_points.append([x_aruco, y_aruco])

        # Giới hạn số điểm để tránh lag
        MAX_POINTS = 1000
        if len(self.desired_points) > MAX_POINTS:
            self.desired_points.pop(0)
        if len(self.actual_points) > MAX_POINTS:
            self.actual_points.pop(0)

        # --- Vẽ quỹ đạo ---
        if not hasattr(self, "desired_line"):
            # Khởi tạo line chỉ một lần
            self.desired_line, = self.ax1.plot([], [], 'r-', label="Desired")
            self.actual_line,  = self.ax1.plot([], [], 'b-', label="Actual")
            self.ax1.set_xlabel("x (cm)")
            self.ax1.set_ylabel("y (cm)")
            self.ax1.legend()

        self.desired_line.set_data([p[0] for p in self.desired_points],
                                [p[1] for p in self.desired_points])
        self.actual_line.set_data([p[0] for p in self.actual_points],
                                [p[1] for p in self.actual_points])

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.plot_widget.draw()

        # --- Error plots ---
        ex = x_aruco - xd
        ey = y_aruco - yd
        e_yaw = yaw_aruco - yaw_d

        if not hasattr(self, "error_lines"):
            self.error_lines = {
                "x": self.ax2.plot([], [], 'r-')[0],
                "y": self.ax3.plot([], [], 'b-')[0],
                "yaw": self.ax4.plot([], [], 'g-')[0],
            }
            self.ax2.set_ylabel("error x (cm)")
            self.ax3.set_ylabel("error y (cm)")
            self.ax4.set_ylabel("error yaw (rad)")

            self.error_data = {"t": [], "x": [], "y": [], "yaw": []}

        # Lưu dữ liệu error
        self.error_data["t"].append(self.t)
        self.error_data["x"].append(ex)
        self.error_data["y"].append(ey)
        self.error_data["yaw"].append(e_yaw)

        # Giới hạn số điểm error
        if len(self.error_data["t"]) > MAX_POINTS:
            for k in self.error_data:
                self.error_data[k].pop(0)

        # Update line
        self.error_lines["x"].set_data(self.error_data["t"], self.error_data["x"])
        self.error_lines["y"].set_data(self.error_data["t"], self.error_data["y"])
        self.error_lines["yaw"].set_data(self.error_data["t"], self.error_data["yaw"])

        for ax in [self.ax2, self.ax3, self.ax4]:
            ax.relim()
            ax.autoscale_view()

        self.plot_widget_1.draw()
        self.plot_widget_2.draw()
        self.plot_widget_3.draw()
        
    def update_frame(self):     
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return

        # ------------------------------ Detect ArUco ------------------------------
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is not None:
            all_marker, distance_infor = [], []
            # Using custom pose estimation function to get rvecs and tvecs
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                # Draw axis for each marker
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size)

                # Convert rotation vector to rotation matrix using cv2.Rodrigues
                rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
                id = ids[i][0]
                if id <= 13:
                    x0, y0, phi0 = marker_start[id]
                    # Calculate global position of the ArUco marker
                    xaruco, yaruco, phiaruco, distance = get_aruco(x0, y0, phi0, rotation_matrix, tvecs[i])
                    phiaruco = normalize_angle_deg(math.degrees(phiaruco))

                    marker_info = [xaruco, yaruco, phiaruco]
                    all_marker.append(marker_info)
                    distance_infor.append(distance)

            if all_marker:
                # Weighted average based on distance from camera to marker
                weights = 1 / (np.array(distance_infor) + 1e-6)
                weights /= np.sum(weights)
                x_est = np.sum(weights * np.array([p[0] for p in all_marker]))
                y_est = np.sum(weights * np.array([p[1] for p in all_marker]))
                phi_est = np.sum(weights * np.array([p[2] for p in all_marker]))

                x_filt, y_filt, phi_filt = LowPassFilter(x_est, y_est, phi_est)
                x_smooth, y_smooth, phi_smooth = smooth_pose(x_filt, y_filt, phi_filt)
                self.x_kf = kf_x.update(x_smooth)
                self.y_kf = kf_y.update(y_smooth)
                self.yaw_kf = kf_yaw.update(phi_smooth)

                pose = {
                    'x': self.x_kf,
                    'y': self.y_kf,
                    'yaw': self.yaw_kf * math.pi / 180
                }
                
                # send_pose(pose['x'], pose['y'], pose['yaw'])

                cv2.putText(frame, f"X={pose['x']:.3f}, Y={pose['y']:.3f}, yaw={pose['yaw']:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Convert frame sang QImage để hiển thị trong QLabel
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        self.Camera.setPixmap(QtGui.QPixmap.fromImage(qimg))

    # ---------------- Send Data to Controller ----------------
    def send_to_ipc(self):
        if not self.running:   # chỉ gửi khi đang chạy
            return
        data = {
            "x_global_aruco": self.x_aruco,
            "y_global_aruco": self.y_aruco,
            "yaw_aruco": self.yaw_aruco,
            "x_global_desired": self.xd,
            "y_global_desired": self.yd,
            "yaw_desired": self.yaw_d,
            "x_dot_global_desired": self.vx_d,
            "y_dot_global_desired": self.vy_d,
            "x_2dot_global_desired": self.ax_d,
            "y_2dot_global_desired": self.ay_d
        }
        try:
            if hasattr(self, "sock"):   # chỉ gửi nếu đã connect
                msg = json.dumps(data) + "\n"
                self.sock.sendall(msg.encode())
        except Exception as e:
            print("Send error:", e)

    # ---------------- Receive Data from Controller ----------------
    def check_socket_data(self):
        try:
            data = self.sock.recv(4096).decode()
            if not data:
                return

            if not hasattr(self, "recv_buffer"):
                self.recv_buffer = ""
            self.recv_buffer += data

            while "\n" in self.recv_buffer:
                line, self.recv_buffer = self.recv_buffer.split("\n", 1)
                try:
                    msg = json.loads(line)
                    self.handle_ipc_data(msg)
                except Exception as e:
                    print("JSON parse error:", e)

        except BlockingIOError:
            pass  # không có dữ liệu mới
        except Exception as e:
            print("Recv error:", e)

    def handle_ipc_data(self, data):
        self.x_EKF  = data.get("x_global_EKF", 0.0)
        self.y_EKF  = data.get("y_global_EKF", 0.0)
        self.yaw_EKF = data.get("yaw_EKF", 0.0)
        self.v_local_encoder = data.get("v_local_encoder", [0.0, 0.0, 0.0])
        self.v_local_EKF     = data.get("v_local_EKF", [0.0, 0.0, 0.0])

        # Cập nhật UI nếu cần
        self.label_x_EKF.setText(f"{self.x_EKF:.2f}")
        self.label_y_EKF.setText(f"{self.y_EKF:.2f}")
        self.label_yaw_EKF.setText(f"{self.yaw_EKF:.2f}")
            
    def closeEvent(self, event):
        # Giải phóng camera khi thoát
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ArucoApp()
    window.show()
    sys.exit(app.exec_())
