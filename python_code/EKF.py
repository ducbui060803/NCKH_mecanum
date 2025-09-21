import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
import time

# --- EKF class ---
def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

class EKFBody:
    def __init__(self, dt, Q_fixed, R_fixed):
        self.dt = dt
        # state: x, y, yaw, vx_body, vy_body
        self.x_t = np.array([[0.0], [0.0], [np.pi/2], [0.0], [0.0]])
        self.P_t = np.diag([0.1, 0.1, 0.1, 0.1, 0.1])
        self.Q_t = np.diag(Q_fixed)
        self.R_t = np.diag(R_fixed)
        self.H_t = np.eye(5)

    def predict(self, u_t):
        omega = float(u_t[0])
        x, y, yaw, vx_body, vy_body = self.x_t.flatten()

        # --- propagate position & yaw ---
        dx = (vx_body * np.cos(yaw) - vy_body * np.sin(yaw)) * self.dt
        dy = (vx_body * np.sin(yaw) + vy_body * np.cos(yaw)) * self.dt
        dyaw = omega * self.dt

        self.x_t[0,0] = x + dx
        self.x_t[1,0] = y + dy
        self.x_t[2,0] = wrap_to_pi(yaw + dyaw)
        self.x_t[3,0] = vx_body
        self.x_t[4,0] = vy_body
        # vx_body, vy_body giữ nguyên

        # Jacobian
        F_t = np.array([
            [1., 0., (-vx_body*np.sin(yaw) - vy_body*np.cos(yaw))*self.dt, self.dt*np.cos(yaw), -self.dt*np.sin(yaw)],
            [0., 1., (vx_body*np.cos(yaw) - vy_body*np.sin(yaw))*self.dt, self.dt*np.sin(yaw),  self.dt*np.cos(yaw)],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.]
        ])
        self.P_t = F_t @ self.P_t @ F_t.T + self.Q_t

    def update(self, z_t):
        z = np.array(z_t, dtype=float).reshape((-1,1))
        y_tilde = z - self.H_t @ self.x_t
        # wrap yaw error
        y_tilde[2,0] = wrap_to_pi(y_tilde[2,0])
        S = self.H_t @ self.P_t @ self.H_t.T + self.R_t
        K = self.P_t @ self.H_t.T @ np.linalg.inv(S)
        self.x_t = self.x_t + K @ y_tilde
        self.x_t[2,0] = wrap_to_pi(self.x_t[2,0])
        self.P_t = (np.eye(5) - K @ self.H_t) @ self.P_t

    def get_state(self):
        return self.x_t.copy()

# --- ROS Node ---
x_aruco = 0.0
y_aruco = 0.0
yaw_aruco = 0.0
vx_local = 0.0
vy_local = 0.0
yaw_dot = 0.0
imu_yaw = 0.0

def uart_callback(msg):
    global vx_local, vy_local, imu_yaw, yaw_dot
    if len(msg.data) >= 4:
        vx_local = msg.data[0]
        vy_local = msg.data[1]
        imu_yaw = msg.data[2]
        yaw_dot = msg.data[3]

def controller_callback(msg):
    global x_aruco, y_aruco, yaw_aruco
    if len(msg.data) >= 3:
        x_aruco = msg.data[0]
        y_aruco = msg.data[1]
        yaw_aruco = msg.data[2]

def publish_pose(pose):
    msg = Float32MultiArray()
    msg.data = [pose['x_EKF'], pose['y_EKF'], pose['yaw_EKF'], pose['vx_body_enc'], pose['vy_body_enc'], pose['yaw_dot']]
    pose_pub.publish(msg)

# --- Main ---
dt = 0.01
Q_fixed = [0.2, 0.2, 0.1, 0.02, 0.02]
R_fixed = [0.005, 0.005, 0.001, 0.01, 0.01]

ekf = EKFBody(dt, Q_fixed, R_fixed)

rospy.init_node("ekf_node", anonymous=True)
pose_pub = rospy.Publisher("/ekf_pose", Float32MultiArray, queue_size=10)
rospy.Subscriber("/uart_data", Float32MultiArray, uart_callback)
rospy.Subscriber("/cmd_vel", Float32MultiArray, controller_callback)

rate = rospy.Rate(100)
time_stamps = []
v_measured_list = []
v_estimated_list = []
start_time = time.time()

while not rospy.is_shutdown():

    # Lấy measurement từ controller node
    x_meas = x_aruco
    y_meas = y_aruco
    yaw_meas = yaw_aruco

    # Fusing yaw với imu
    yaw_meas = wrap_to_pi(0.4*yaw_meas + 0.6*imu_yaw)

    # Chuẩn bị u_t
    u_t = np.array([yaw_dot])

    # Measurement vector z_t = [x, y, yaw, vx_body, vy_body]
    z_t = [x_meas, y_meas, yaw_meas, vx_local, vy_local]

    # EKF
    ekf.predict(u_t)
    ekf.update(z_t)

    state = ekf.get_state().flatten()

    # Prepare pose dictionary
    pose = {
        'x_EKF': state[0],
        'y_EKF': state[1],
        'yaw_EKF': state[2],
        'vx_body_enc': state[3],
        'vy_body_enc': state[4],
        'yaw_dot': yaw_dot,
    }

    publish_pose(pose)

    rate.sleep()