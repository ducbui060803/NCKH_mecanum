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
    plt.plot(time_stamp, yaw_desired_list, 'g--', label="yaw_desired", linewidth=0.8)
    plt.plot(time_stamp, yaw_dist_list, 'r-', label="yaw_dist ", linewidth=0.8)
    plt.plot(time_stamp, yaw_ekf_list, 'b-', label="yaw_ekf ", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("yaw (degree)")
    plt.title("Comparison of yaw_ekf vs yaw_dist")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_x_ekf():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamp, x_desired_list, 'g--', label="x_desired", linewidth=0.8)
    plt.plot(time_stamp, x_dist_list, 'r-', label="x_dist ", linewidth=0.8)
    plt.plot(time_stamp, x_ekf_list, 'b-', label="x_ekf ", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("x (m)")
    plt.title("Comparison of x_ekf vs x_dist")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_y_ekf():
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamp, y_desired_list, 'g--', label="y_desired", linewidth=0.8)
    plt.plot(time_stamp, y_dist_list, 'r-', label="y_dist ", linewidth=0.8)
    plt.plot(time_stamp, y_ekf_list, 'b-', label="y_ekf ", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("y (m)")
    plt.title("Comparison of y_ekf vs y_desired")
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
def wrap_to_pi(angle):
    # return np.arctan2(np.sin(angle), np.cos(angle))
    return angle
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
        if H is None:
            H = self.H_t
        if R is None:
            R = self.R_t
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

x_cam = add_noise(np.array(x_gt), 0.05)     # camera noise
y_cam = add_noise(np.array(y_gt), 0.05)
phi_cam = add_noise(np.array(phi_gt), 0.05)
v_enc = add_noise(np.array(v_gt), 0.05)     # encoder noise
yaw_dot_imu = add_noise(np.ones(len(time_stamps)) * 0.001, 0.01)  # rad/s

prev_v_enc = v_enc[0]

for k in range(len(time_stamps)):
    # Input từ encoder + IMU
    
    curr_v_enc = v_enc[k]
    delta_v = curr_v_enc - prev_v_enc
    omega = float(yaw_dot_imu[k])    # imu yaw rate
    
    u_t = np.array([omega, delta_v])  
    
    # Đo từ camera
    z_t = np.array([[x_cam[k]], 
                    [y_cam[k]], 
                    [phi_cam[k]]])
    
    ekf.predict(u_t)
    ekf.update(z_t)
    
    state = ekf.get_state().flatten()
    
    prev_v_enc = curr_v_enc
    
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


plot_x_ekf()
plot_y_ekf()
plot_yaw_ekf()

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