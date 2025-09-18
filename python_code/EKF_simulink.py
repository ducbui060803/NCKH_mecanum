import numpy as np
import matplotlib.pyplot as plt
import time
import math
dt = 0.01
T = 70
time_stamps = np.arange(0, T, dt)

x_ekf_list  = []
x_dist_list  = []
y_ekf_list  = []
y_dist_list  = []
yaw_ekf_list  = []
yaw_dist_list  = []

x_ekf_error = []
x_dist_error = []
y_ekf_error = []
y_dist_error = []
yaw_ekf_error = []
yaw_dist_error = []
    
x_ekf_error_mean = 0
x_dist_error_mean = 0
y_ekf_error_mean = 0
y_ekf_error_mean = 0
yaw_ekf_error_mean = 0
yaw_dist_error_mean = 0
x_ekf_error_max = 0
x_dist_error_max = 0
y_ekf_error_max = 0
y_dist_error_max = 0
yaw_ekf_error_max = 0
yaw_dist_error_max = 0

yaw_desired_list = []
x_desired_list = []
y_desired_list = []
time_stamp = []
start_time = time.time()

def plot_yaw_ekf():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamp, yaw_ekf_list, 'b-', label="yaw_ekf ", linewidth=0.8)
    plt.plot(time_stamp, yaw_desired_list, 'r--', label="yaw_desired", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("yaw (degree)")
    plt.title("Comparison of yaw_ekf vs yaw_desired")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_yaw_dist():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamp, yaw_dist_list, 'b-', label="yaw_dist ", linewidth=0.8)
    plt.plot(time_stamp, yaw_desired_list, 'r--', label="yaw_desired", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("yaw (degree)")
    plt.title("Comparison of yaw_dist vs yaw_desired")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_x_ekf():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamp, x_ekf_list, 'g-', label="x_ekf ", linewidth=0.8)
    plt.plot(time_stamp, x_desired_list, 'r--', label="x_desired", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("x (m)")
    plt.title("Comparison of x_ekf vs x_desired")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_x_dist():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamp, x_dist_list, 'g-', label="x_dist ", linewidth=0.8)
    plt.plot(time_stamp, x_desired_list, 'r--', label="x_desired", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("x(m)")
    plt.title("Comparison of x_dist vs x_desired")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_y_ekf():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamp, y_ekf_list, '-', color='purple', label="y_ekf ", linewidth=0.8)
    plt.plot(time_stamp, y_desired_list, 'r--', label="y_desired", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("y (m)")
    plt.title("Comparison of y_ekf vs y_desired")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_y_dist():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamp, y_dist_list, '-', color='purple', label="y_dist ", linewidth=0.8)
    plt.plot(time_stamp, y_desired_list, 'r--', label="y_desired", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("y (m)")
    plt.title("Comparison of y_dist vs y_desired")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Quỹ đạo ground truth: đi thẳng theo x
x_gt, y_gt, phi_gt, v_gt = [], [], [], []
est_x, est_y, est_phi = [], [], []
x, y, phi, v = 0, 0, 0, 0.1  # v = 0.1 m/s


# process noise for [x, y, phi, v]
# Q_fixed = [1e-4, 1e-4, 1e-6, 1e-3]
# measurement noise for [x_meas, y_meas, phi_meas]
# R_fixed = [0.03**2, 0.03**2, (8.0*np.pi/180)**2]  # sigma_x=2cm, sigma_phi=1deg
# Q_fixed = [0.001, 0.001, 0.0001, 0.0001]
# R_fixed = [0.1, 0.1, 0.00001]
Q_fixed = [1e-4, 1e-4, 5e-6, 1e-3] # x~1cm, y~1cm, phi~0.001 rad, v~0.03 m/s
R_fixed = [ (0.03)**2, (0.03)**2, (np.deg2rad(1.0))**2 ] # -> sigma_x = 3 cm, sigma_phi = 1 degree

pre_v = 0
acc_prev = 0
yaw_dot = 0
imu_yaw = 0

def _wrap_angle(a):
    # normalize to [-pi, pi)
    return math.atan2(math.sin(a), math.cos(a))
# --- Extended Kalman Filter ---
class EKF:
    def __init__(self, dt, Q_fixed, R_fixed):
        """ Khởi tạo EKF với ma trận nhiễu Q và R cố định """
        self.dt = dt  # Bước thời gian (delta_t)

        # Trạng thái hệ thống x_t = [x, y, phi, v]
        self.x_t = np.array([
                            [0.0],         # x
                            [0.0],        # y
                            # [np.pi/2 ],   # phi
                            [0.0 ],
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
        # acc = 0 * acc_prev + 1 * (u_t[0] - pre_v)
        # Ma trận điều khiển B_t
        B_t = np.array([
                [np.cos(yaw_t) * dt * u_t[0]],
                [np.sin(yaw_t) * dt * u_t[0]],
                [u_t[1]],
                # [u_t[0]-pre_v]
                # [acc]
                [0.0]
            ])
        
        # Dự đoán trạng thái mới
        self.x_t = self.x_t + B_t  # x_t|t-1 = f(x_t-1, u_t)


        acc_prev = acc
        self.P_t = F_t @ self.P_t @ F_t.T + self.Q_t  # Cập nhật ma trận hiệp phương sai P_t|t-1
        # self.P_t = F_t @ self.P_t @ F_t.T  # Cập nhật ma trận hiệp phương sai P_t|t-1

        # self.x_t[2, 0] = np.arctan2(np.sin(self.x_t[2, 0]), np.cos(self.x_t[2, 0]))

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

        # Cập nhật trạng thái
        self.x_t = self.x_t + K_t @ d_t

        self.P_t = (np.eye(4) - K_t @ self.H_t) @ self.P_t
        # self.x_t[2, 0] = np.arctan2(np.sin(self.x_t[2, 0]), np.cos(self.x_t[2, 0]))


    def get_state(self):
        """ Trả về trạng thái hiện tại """
        return self.x_t
    
ekf = EKF(dt, Q_fixed, R_fixed)

def trajectory_line(dt, T, v=0.1):
    """Quỹ đạo đi thẳng"""
    x, y, phi = 0, 0, 0
    x_gt, y_gt, phi_gt, v_gt = [], [], [], []
    time_stamps = np.arange(0, T, dt)
    for t in time_stamps:
        x += v * np.cos(phi) * dt
        y += v * np.sin(phi) * dt
        phi += 0 * dt
        x_gt.append(x)
        y_gt.append(y)
        phi_gt.append(phi)
        v_gt.append(v)
    return np.array(x_gt), np.array(y_gt), np.array(phi_gt), np.array(v_gt)

def trajectory_circle(dt, T, v=0.1, R=1.0):
    """Quỹ đạo hình tròn bán kính R"""
    x, y, phi = R, 0, np.pi/2
    x_gt, y_gt, phi_gt, v_gt = [], [], [], []
    time_stamps = np.arange(0, T, dt)
    omega = v / R
    for t in time_stamps:
        phi += omega * dt
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        x_gt.append(x)
        y_gt.append(y)
        phi_gt.append(phi)
        v_gt.append(v)
    return np.array(x_gt), np.array(y_gt), np.array(phi_gt), np.array(v_gt)

def trajectory_square(dt, T, v=0.1, L=1.0):
    """Quỹ đạo hình vuông cạnh L"""
    x, y, phi = 0, 0, 0
    x_gt, y_gt, phi_gt, v_gt = [], [], [], []
    time_stamps = np.arange(0, T, dt)
    side_time = int(L / (v * dt))
    directions = [0, np.pi/2, np.pi, -np.pi/2]  # 4 cạnh
    d = 0
    for i, t in enumerate(time_stamps):
        phi = directions[d % 4]
        x += v * np.cos(phi) * dt
        y += v * np.sin(phi) * dt
        if (i+1) % side_time == 0:
            d += 1  # đổi hướng khi hết cạnh
        x_gt.append(x)
        y_gt.append(y)
        phi_gt.append(phi)
        v_gt.append(v)
    return np.array(x_gt), np.array(y_gt), np.array(phi_gt), np.array(v_gt)

def trajectory_figure8(dt, T, a=1.0, b=0.5, w=0.2):
    """Quỹ đạo số 8 theo phương trình Lissajous"""
    x_gt, y_gt, phi_gt, v_gt = [], [], [], []
    time_stamps = np.arange(0, T, dt)
    for t in time_stamps:
        x = a * np.sin(w * t)
        y = b * np.sin(2 * w * t)
        dx = a * w * np.cos(w * t)
        dy = 2 * b * w * np.cos(2 * w * t)
        phi = np.pi/2
        v = np.sqrt(dx**2 + dy**2)
        x_gt.append(x)
        y_gt.append(y)
        phi_gt.append(phi)
        v_gt.append(v)
    return np.array(x_gt), np.array(y_gt), np.array(phi_gt), np.array(v_gt)

# Sinh noise cho sensor
def add_noise(data, sigma):
    return data + np.random.normal(0, sigma, size=len(data))

# x_gt, y_gt, phi_gt, v_gt = trajectory_figure8(dt, T)
# Hoặc quỹ đạo khác:
# x_gt, y_gt, phi_gt, v_gt = trajectory_line(dt, T)
# x_gt, y_gt, phi_gt, v_gt = trajectory_circle(dt, T, R=1.0)
x_gt, y_gt, phi_gt, v_gt = trajectory_square(dt, T, L=1.0)

x_cam = add_noise(np.array(x_gt), 0.03)     # camera noise
y_cam = add_noise(np.array(y_gt), 0.03)
phi_cam = add_noise(np.array(phi_gt), 0.03)
v_enc = add_noise(np.array(v_gt), 0.01)     # encoder noise
# yaw_dot_imu = add_noise(np.zeros(len(time_stamps)), 0.005)  # IMU noise
yaw_dot_imu = add_noise(np.ones(len(time_stamps)) * 0.001, 0.005)  # rad/s
for k in range(len(time_stamps)):
    # Input từ encoder + IMU
    u_t = np.array([v_enc[k], yaw_dot_imu[k]])  
    
    # Đo từ camera
    z_t = np.array([[x_cam[k]], 
                    [y_cam[k]], 
                    [phi_cam[k]]])
    
    ekf.predict(u_t)
    ekf.update(z_t)
    
    state = ekf.get_state().flatten()
    est_x.append(state[0])
    est_y.append(state[1])
    est_phi.append(state[2])
    
    x_ekf_list.append(state[0])
    y_ekf_list.append(state[1])
    yaw_ekf_list.append(state[2])
    x_dist_list.append(x_cam[k])
    y_dist_list.append(y_cam[k])
    yaw_dist_list.append(phi_cam[k])
    x_desired_list.append(x_gt[k])
    y_desired_list.append(y_gt[k])
    yaw_desired_list.append(phi_gt[k])
    
    x_ekf_error.append(abs(state[0]-x_gt[k]))
    x_dist_error.append(abs(x_cam[k]-x_gt[k]))
    y_ekf_error.append(abs(state[1]-y_gt[k]))
    y_dist_error.append(abs(y_cam[k]-y_gt[k]))
    yaw_ekf_error.append(abs(state[2]-phi_gt[k]))
    yaw_dist_error.append(abs(phi_cam[k]-phi_gt[k]))

    time_stamp.append(time_stamps[k])
    

# Vẽ kết quả
plt.plot(x_gt, y_gt, 'r--', label="Ground Truth")
plt.plot(x_cam, y_cam, 'g-', linewidth=0.5, label="Camera noisy")
plt.plot(est_x, est_y, 'b-', linewidth=0.5, label="EKF estimate")
plt.legend()
plt.axis("equal")
plt.show()

plot_yaw_ekf()
plot_yaw_dist()
plot_x_ekf()
plot_x_dist()
plot_y_ekf()
plot_y_dist()

yaw_ekf_error_mean = sum(yaw_ekf_error)/len(yaw_ekf_error)
yaw_dist_error_mean = sum(yaw_dist_error)/len(yaw_dist_error)
x_ekf_error_mean = sum(x_ekf_error)/len(x_ekf_error)
x_dist_error_mean = sum(x_dist_error)/len(x_dist_error)
y_ekf_error_mean = sum(y_ekf_error)/len(y_ekf_error)
y_dist_error_mean = sum(y_dist_error)/len(y_dist_error)

yaw_ekf_error_max = max(yaw_ekf_error)
yaw_dist_error_max = max(yaw_dist_error)
x_ekf_error_max = max(x_ekf_error)
x_dist_error_max = max(x_dist_error)
y_ekf_error_max = max(y_ekf_error)
y_dist_error_max = max(y_dist_error)

print('----------------------------------')
print(f'x_ekf_error_max: {x_ekf_error_max}')
print(f'x_dist_error_max: {x_dist_error_max}')
print(f'y_ekf_error_max: {y_ekf_error_max}')
print(f'y_dist_error_max: {y_dist_error_max}')
print(f'yaw_ekf_error_max: {yaw_ekf_error_max}')
print(f'yaw_dist_error_max: {yaw_dist_error_max}')
print('----------------------------------')
print(f'x_ekf_error_mean: {x_ekf_error_mean}')
print(f'x_dist_error_mean: {x_dist_error_mean}')
print(f'y_ekf_error_mean: {y_ekf_error_mean}')
print(f'y_dist_error_mean: {y_dist_error_mean}')
print(f'yaw_ekf_error_mean: {yaw_ekf_error_mean}')
print(f'yaw_dist_error_mean: {yaw_dist_error_mean}')
print('----------------------------------')