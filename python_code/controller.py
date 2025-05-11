
import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from scipy.integrate import solve_ivp

theta_prev = 0
threshold = 0.1

# Tham số hệ thống
m = 7           # Khối lượng robot (kg)
I_z = 0.22 + 0.02328     # Mô-men quán tính quanh trục z (Xe + banh xe) (kg.m^2)
r = 0.02        # Bán kính bánh xe (m)
L1 = 0.15       # Khoảng cách từ tâm đến bánh xe (m)
L2 = 0.08       # Khoảng cách từ tâm đến bánh xe (m)

# Ma trận quán tính
D = lambda: np.array([
    [m,         0,          -m * L1],
    [0,         m,          m * L2],
    [-m * L1,   m * L2,     I_z + m * (L2**2 + L1**2)]
]) 

class BacksteppingController:
    # Vector n(ζ)
    def n_zeta(self, u, v, r):
        return np.array([
            -m * (v + L2 * r),
            m * (u - L1 * r),
            m * r * (L2 * u + L1 * v)
        ])
    
    def __init__(self):
        rospy.init_node('backstepping_mecanum_controller', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        #rospy.Subscriber('/pose_data', Float32MultiArray, self.odom_callback)
        #rospy.Subscriber('/uart_data', Float32MultiArray, self.odom_callback)
        # Controller gains
        self.k1 = 1.5
        self.k2 = 1.2

        # Robot state
        self.x = 0.0
        self.y = -1.0
        self.theta = math.pi/2
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.theta_dot = 0.0

        # Desired state (mục tiêu)
        self.x_d = 1.0  
        self.y_d = 1.0
        self.theta_d = math.pi/2 
        self.x_dot_d = 0.0  
        self.y_dot_d = 0.0
        self.theta_dot_d = 0.0

        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.spin()
    
    def odom_callback(self, msg):
        """ Update robot's state from Float32MultiArray. """
        # Lấy giá trị từ message Float32MultiArray
        self.x_d = msg.data[0]  # pose['x']
        self.y_d = msg.data[1]  # pose['y']
        self.theta_d = msg.data[2]  # pose['phi']
        self.v = msg.data[3]
        self.x_dot_d = self.v * math.sin(self.theta_d)
        self.x_dot_d = self.v * math.cos(self.theta_d) 
    def uart_callback(self, msg):
        self.theta_dot_d = msg.data[1]
    
    def control_loop(self, event):
        """ Apply backstepping control. """

        # Tính toán các giá trị tốc độ mong muốn
        self.x_dot_d = -self.r * self.omega * np.sin(self.theta_d * event.current_real.to_sec())
        self.y_dot_d = self.r * self.omega * np.cos(self.theta_d * event.current_real.to_sec())
        self.theta_dot_d = self.omega
    
        e1 = np.array([self.x_d - self.x, self.y_d - self.y, self.theta_d - self.theta])
        e2 = np.array([self.x_dot_d - self.x_dot, self.y_dot_d - self.y_dot, self.theta_dot_d - self.theta_dot])

        # Velocity predict
        eta_dot_d = np.array([self.x_dot_d, self.y_dot_d, self.theta_dot_d])
        v_d = eta_dot_d + self.k1 * e1
        v_d[0:2] = np.clip(v_d[0:2], -0.5, 0.5)
        v_d[2] = np.clip(v_d[2], -1, 1)

        # Áp dụng ma trận nghịch đảo Jacobian
        J = np.array([[np.cos(self.theta), -np.sin(self.theta),     0],
                      [np.sin(self.theta),  np.cos(self.theta),     0],
                      [0,                   0,                      1]])
        J_inv = np.linalg.pinv(J)  # Dùng pseudo-inverse thay vì nghịch đảo
        
        u = np.dot(D(), (self.k2 * e2 + np.dot(J_inv, v_d) + np.dot(J.T, e1))) + self.n_zeta(self.x_dot, self.y_dot, self.theta_dot)
        u[0:2] = np.clip(u[0:2], -0.07, 0.07)  # Giới hạn u[0] và u[1] từ -0.1 đến 0.1
        u[2] = np.clip(u[2], -1, 1)      # Giới hạn riêng cho u[2] từ -0.2 đến 0.2

        # Publish command
        cmd = Twist()
        cmd.linear.x = u[0]
        cmd.linear.y = u[1]
        cmd.angular.z = u[2]
        self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    try:   
        BacksteppingController()
    except rospy.ROSInterruptException:
        pass
