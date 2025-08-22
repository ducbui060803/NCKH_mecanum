#!/usr/bin/env python3
"""
mecanum_ui.py

Single-file PySide6 UI for visualizing a Mecanum robot system.
Features:
- Live camera view from OpenCV (device index or video URL).
- Optional ROS1 integration: subscribe to /camera/image_raw (sensor_msgs/Image) and /pose (geometry_msgs/PoseStamped) if available.
- Trajectory selection (predefined simple trajectories) and reference vs actual plotting (x,y).
- Real-time display of x, y, phi values and position errors (x_err, y_err, yaw_err).
- Uses pyqtgraph for fast plotting.

Dependencies:
- Python 3.8+
- PySide6
- opencv-python
- pyqtgraph
- numpy
- (optional) rospy, cv_bridge for ROS integration

Run (non-ROS):
    python mecanum_ui.py --camera 0

Run (with ROS):
    source /opt/ros/<distro>/setup.bash
    python mecanum_ui.py --ros

This file is meant as a starting point — adapt topics, frame IDs and message processing to your system.

"""

import sys
import argparse
import time
import threading
import math
from collections import deque

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from pyqtgraph import PlotWidget
# Try to import ROS if requested at runtime; keep optional to allow running without ROS
try:
    import rospy
    from sensor_msgs.msg import Image as RosImage
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge
    HAVE_ROS = True
except Exception:
    HAVE_ROS = False


# --------------------------- Worker Threads ---------------------------

class CameraThread(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray)

    def __init__(self, source=0, use_ros=False, ros_topic='/camera/image_raw'):
        super().__init__()
        self.source = source
        self.use_ros = use_ros
        self.ros_topic = ros_topic
        self._running = False
        self.bridge = CvBridge() if HAVE_ROS else None

    def run(self):
        self._running = True
        if self.use_ros and HAVE_ROS:
            # ROS mode: spin in this thread
            rospy.Subscriber(self.ros_topic, RosImage, self._ros_cb)
            rospy.spin()
        else:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"[CameraThread] Cannot open camera source: {self.source}")
                return
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue
                # convert BGR -> RGB for Qt
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)
                time.sleep(0.02)
            cap.release()

    def _ros_cb(self, msg):
        if not self._running:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(frame_rgb)
        except Exception as e:
            print('[CameraThread] ROS image conversion error:', e)

    def stop(self):
        self._running = False
        try:
            self.quit()
            self.wait(2000)
        except Exception:
            pass


class PoseThread(QtCore.QThread):
    pose_ready = QtCore.Signal(float, float, float)  # x, y, yaw (rad)

    def __init__(self, use_ros=False, pose_topic='/pose'):
        super().__init__()
        self.use_ros = use_ros
        self.pose_topic = pose_topic
        self._running = False
        self.bridge = CvBridge() if HAVE_ROS else None

    def run(self):
        self._running = True
        # if self.use_ros and HAVE_ROS:
        #     rospy.Subscriber(self.pose_topic, PoseStamped, self._ros_cb)
        #     rospy.spin()
        # else:
        # Fake pose generator for demo
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            x = 0.5 * math.cos(0.6 * t)
            y = 0.5 * math.sin(0.4 * t)
            yaw = math.atan2(y, x)
            self.pose_ready.emit(x, y, yaw)
            time.sleep(0.05)

    def _ros_cb(self, msg):
        if not self._running:
            return
        x = msg.pose.position.x
        y = msg.pose.position.y
        # extract yaw from quaternion
        q = msg.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.pose_ready.emit(x, y, yaw)

    def stop(self):
        self._running = False
        try:
            self.quit()
            self.wait(2000)
        except Exception:
            pass


# --------------------------- Main UI ---------------------------

class MecanumUI(QtWidgets.QMainWindow):
    def __init__(self, camera_source = 0, use_ros = False):
        super().__init__()
        self.setWindowTitle('Mecanum Control UI')
        self.resize(1500, 800)
        self.setStyleSheet("background-color: white; color: black;")
        self.use_ros = use_ros

        # central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: video + controls
        left_vbox = QtWidgets.QVBoxLayout()
        layout.addLayout(left_vbox, 2)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setStyleSheet("color: white; background: black;")
        left_vbox.addWidget(self.video_label)

        ctrl_h = QtWidgets.QHBoxLayout()
        left_vbox.addLayout(ctrl_h)
        self.traj_combo = QtWidgets.QComboBox()
        self.traj_combo.addItems(["Circle", "Square", "Elipse", "Figure-8"])
        ctrl_h.addWidget(QtWidgets.QLabel('Trajectory:'))
        ctrl_h.addWidget(self.traj_combo)

        self.start_btn = QtWidgets.QPushButton('Start')
        self.stop_btn = QtWidgets.QPushButton('Stop')
        for btn in (self.start_btn, self.stop_btn):
            btn.setStyleSheet("color: black; background-color: #f0f0f0;")
            ctrl_h.addWidget(btn)

        # Numeric displays
        grid = QtWidgets.QGridLayout()
        left_vbox.addLayout(grid)
        labels = ['x (m)', 'y (m)', 'phi (deg)', 'err_x (m)', 'err_y (m)', 'err_phi (deg)']
        self.value_displays = {}
        for i, name in enumerate(labels):
            lbl = QtWidgets.QLabel(name)
            val = QtWidgets.QLabel('0.000')
            val.setMinimumWidth(80)
            val.setStyleSheet('font-weight: bold;')
            grid.addWidget(lbl, i//3, (i%3)*2)
            grid.addWidget(val, i//3, (i%3)*2+1)
            self.value_displays[name] = val

        # Right: plots
        right_vbox = QtWidgets.QVBoxLayout()
        layout.addLayout(right_vbox, 2)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.pw = pg.PlotWidget(title='Trajectory (XY)')
        self.pw.setLabel('left', 'Y (m)')
        self.pw.setLabel('bottom', 'X (m)')
        self.pw.addLegend(offset=(10,10))
        right_vbox.addWidget(self.pw, 2)

        self.ref_curve = self.pw.plot([], [], pen=pg.mkPen(width=2, style=QtCore.Qt.DashLine, color='b'), name='Reference')
        self.actual_curve = self.pw.plot([], [], pen=pg.mkPen(width=2, color='r'), name='Actual')

        self.err_plot = pg.PlotWidget(title='Errors over time')
        self.err_plot.addLegend()
        right_vbox.addWidget(self.err_plot, 1)
        self.err_x_curve = self.err_plot.plot([], [], name='err_x', pen=pg.mkPen(color='r'))
        self.err_y_curve = self.err_plot.plot([], [], name='err_y', pen=pg.mkPen(color='g'))
        self.err_phi_curve = self.err_plot.plot([], [], name='err_phi', pen=pg.mkPen(color='b'))

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Buffers real-time
        self.ref_traj = []
        self.actual_traj = deque(maxlen=2000)
        self.err_time = deque(maxlen=1000)
        self.err_x = deque(maxlen=1000)
        self.err_y = deque(maxlen=1000)
        self.err_phi = deque(maxlen=1000)
        self.start_time = None

        # Buffers offline (Matplotlib)
        self.time_list = []
        self.x_list = []
        self.y_list = []
        self.yaw_list = []
        self.u1_list = []
        self.u2_list = []
        self.u3_list = []
        self.vx_list = []
        self.vy_list = []
        self.omega_list = []

        # Camera/Pose thread
        self.cam_thread = CameraThread(source=camera_source, use_ros=self.use_ros)
        self.cam_thread.frame_ready.connect(self.on_frame)
        self.pose_thread = PoseThread(use_ros=self.use_ros)
        self.pose_thread.pose_ready.connect(self.on_pose)

        # Signals
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.traj_combo.currentIndexChanged.connect(self.on_traj_change)

        self.on_traj_change(0)

        self.ui_timer = QtCore.QTimer()
        self.ui_timer.timeout.connect(self.update_plots)
        self.ui_timer.start(100)

        self.latest_pose = (0.0, 0.0, 0.0)

    def on_start(self):
        self.status.showMessage('Starting...')
        self.start_time = time.time()
        if not self.cam_thread.isRunning():
            self.cam_thread.start()
        if not self.pose_thread.isRunning():
            self.pose_thread.start()
        self.status.showMessage('Running')

    def on_stop(self):
        self.status.showMessage('Stopping...')
        self.cam_thread.stop()
        self.pose_thread.stop()
        self.status.showMessage('Stopped')
        self.plot_matplotlib()

    def on_traj_change(self, idx):
        t = np.linspace(0, 20, 400)
        if idx == 0:
            x = 0.6 * np.cos(0.6 * t)
            y = 0.6 * np.sin(0.6 * t)
        elif idx == 1:
            x = 0.05 * t - 0.5
            y = np.zeros_like(t)
        elif idx == 2:
            x = np.zeros_like(t)
            y = 0.05 * t - 0.5
        elif idx == 3:
            x = 0.6 * np.sin(0.6 * t)
            y = 0.4 * np.sin(1.2 * t)
        else:
            x = np.zeros_like(t)
            y = np.zeros_like(t)
        self.ref_traj = list(zip(x, y))
        self.ref_curve.setData(x.tolist(), y.tolist())
        self.pw.enableAutoRange()
        self.err_plot.enableAutoRange()

    def on_frame(self, frame_rgb):
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(image).scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def on_pose(self, x, y, yaw, u1=0, u2=0, u3=0, vx=0, vy=0, omega=0):
        self.latest_pose = (x, y, yaw)
        self.actual_traj.append((x, y))

        if len(self.ref_traj) > 0:
            ref_x, ref_y = min(self.ref_traj, key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)
        else:
            ref_x, ref_y = x, y
        err_x = ref_x - x
        err_y = ref_y - y
        err_phi = (0.0 - yaw)

        stamp = time.time() - (self.start_time or time.time())
        self.err_time.append(stamp)
        self.err_x.append(err_x)
        self.err_y.append(err_y)
        self.err_phi.append(math.degrees(err_phi))

        # Save offline data
        self.time_list.append(stamp)
        self.x_list.append(x)
        self.y_list.append(y)
        self.yaw_list.append(math.degrees(yaw))
        self.u1_list.append(u1)
        self.u2_list.append(u2)
        self.u3_list.append(u3)
        self.vx_list.append(vx)
        self.vy_list.append(vy)
        self.omega_list.append(omega)

        self.value_displays['x (m)'].setText(f"{x:.3f}")
        self.value_displays['y (m)'].setText(f"{y:.3f}")
        self.value_displays['phi (deg)'].setText(f"{math.degrees(yaw):.2f}")
        self.value_displays['err_x (m)'].setText(f"{err_x:.3f}")
        self.value_displays['err_y (m)'].setText(f"{err_y:.3f}")
        self.value_displays['err_phi (deg)'].setText(f"{math.degrees(err_phi):.2f}")

    def update_plots(self):
        if len(self.actual_traj) > 0:
            xs, ys = zip(*self.actual_traj)
            self.actual_curve.setData(xs, ys)
        if len(self.err_time) > 0:
            self.err_x_curve.setData(list(self.err_time), list(self.err_x))
            self.err_y_curve.setData(list(self.err_time), list(self.err_y))
            self.err_phi_curve.setData(list(self.err_time), list(self.err_phi))

    def plot_matplotlib(self):
        # 1. Quỹ đạo XY
        plt.figure()
        plt.plot(self.x_list, self.y_list, 'r', label='Actual')
        ref_xs, ref_ys = zip(*self.ref_traj)
        plt.plot(ref_xs, ref_ys, 'b--', label='Reference')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.title('Trajectory XY')

        # 2. Sai số
        plt.figure()
        plt.plot(self.time_list, self.err_x, 'r', label='err_x')
        plt.plot(self.time_list, self.err_y, 'g', label='err_y')
        plt.plot(self.time_list, self.err_phi, 'b', label='err_phi')
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Errors over time')

        # 3. X, Y, yaw
        plt.figure()
        plt.plot(self.time_list, self.x_list, label='x')
        plt.plot(self.time_list, self.y_list, label='y')
        plt.plot(self.time_list, self.yaw_list, label='yaw (deg)')
        plt.xlabel('Time (s)')
        plt.ylabel('Pose')
        plt.legend()
        plt.title('Pose over time')

        # 4. U1, U2, U3
        plt.figure()
        plt.plot(self.time_list, self.u1_list, label='u1')
        plt.plot(self.time_list, self.u2_list, label='u2')
        plt.plot(self.time_list, self.u3_list, label='u3')
        plt.xlabel('Time (s)')
        plt.ylabel('Control signal')
        plt.legend()
        plt.title('Control signals')

        # 5. Vx, Vy, Omega
        plt.figure()
        plt.plot(self.time_list, self.vx_list, label='vx')
        plt.plot(self.time_list, self.vy_list, label='vy')
        plt.plot(self.time_list, self.omega_list, label='omega')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.legend()
        plt.title('Velocities over time')

        plt.show()

    def closeEvent(self, event):
        try:
            self.cam_thread.stop()
            self.pose_thread.stop()
        except Exception:
            pass
        event.accept()


# --------------------------- CLI & main ---------------------------

def main():
    parser = argparse.ArgumentParser(description='Mecanum control UI')
    parser.add_argument('--camera', '-c', default=0, help='camera source (device index or URL)')
    parser.add_argument('--ros', action='store_true', help='enable ROS subscribers (requires rospy and cv_bridge)')
    args = parser.parse_args()

    # if ROS requested but not available -> warn
    if args.ros and not HAVE_ROS:
        print('[Warning] ROS support requested but rospy/cv_bridge not available. Running in non-ROS demo mode.')

    # If camera arg is numeric string, convert to int
    cam_src = args.camera
    try:
        cam_src = int(cam_src)
    except Exception:
        cam_src = args.camera

    # init qt
    app = QtWidgets.QApplication(sys.argv)

    # if ROS mode requested and available, initialize rospy in a separate thread
    if args.ros and HAVE_ROS:
        rospy.init_node('mecanum_ui_node', anonymous=True)

    win = MecanumUI(camera_source=cam_src, use_ros=(args.ros and HAVE_ROS))
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
