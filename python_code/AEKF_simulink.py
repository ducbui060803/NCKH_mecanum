import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
T = 5
time_stamps = np.arange(0, T, dt)

# Quỹ đạo ground truth: đi thẳng theo x
x_gt, y_gt, phi_gt, v_gt = [], [], [], []
est_x, est_y, est_phi = [], [], []
x, y, phi, v = 0, 0, 0, 0.1  # v = 0.1 m/s

alpha_Q = 0.9  # Hệ số quên cho nhiễu quá trình
alpha_R = 1  # Hệ số quên cho nhiễu đo lường
Q_fixed = [0.1, 0.1, 0.1, 0.02]
R_fixed = [0.5, 0.5, 0.5]  # Ma trận nhiễu đo lường cố định
pre_v = 0
acc_prev = 0
yaw_dot = 0
imu_yaw = 0
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

for t in time_stamps:
    x += v * np.cos(phi) * dt
    y += v * np.sin(phi) * dt
    phi += 0 * dt  # không quay
    x_gt.append(x)
    y_gt.append(y)
    phi_gt.append(phi)
    v_gt.append(v)

# Sinh noise cho sensor
def add_noise(data, sigma):
    return data + np.random.normal(0, sigma, size=len(data))

x_cam = add_noise(np.array(x_gt), 0.00)     # camera noise
y_cam = add_noise(np.array(y_gt), 0.01)
phi_cam = add_noise(np.array(phi_gt), 0.00)
v_enc = add_noise(np.array(v_gt), 0.01)     # encoder noise
yaw_dot_imu = add_noise(np.zeros(len(time_stamps)), 0.005)  # IMU noise

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

# Vẽ kết quả
plt.plot(x_gt, y_gt, 'r--', label="Ground Truth")
plt.plot(x_cam, y_cam, 'g-', linewidth=0.5, label="Camera noisy")
plt.plot(est_x, est_y, 'b-', linewidth=0.5, label="EKF estimate")
plt.legend()
plt.axis("equal")
plt.show()