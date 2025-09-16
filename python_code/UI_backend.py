# main.py
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from gui_ui import Ui_MainWindow

class MainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Gán sự kiện
        self.Start_btn.clicked.connect(self.start_plot)
        self.Stop_btn.clicked.connect(self.stop_plot)
        self.Clear_Graph_btn_.clicked.connect(self.clear_plot)

        # Timer update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Dữ liệu
        self.t = 0
        self.x_data, self.y_data, self.yaw_data = [], [], []
        self.x_ref, self.y_ref = [], []

    def start_plot(self):
        path = self.comboBox.currentText()
        if path == "Square":
            self.x_ref = [0, 1, 1, 0, 0]
            self.y_ref = [0, 0, 1, 1, 0]
        elif path == "Circle":
            theta = np.linspace(0, 2*np.pi, 100)
            self.x_ref = np.cos(theta)
            self.y_ref = np.sin(theta)
        else:
            self.x_ref, self.y_ref = [], []

        self.ax1.clear()
        self.ax1.plot(self.x_ref, self.y_ref, 'r--', label="Reference Path")
        self.ax1.legend()
        self.plot_widget.draw()

        self.timer.start(100)  # update mỗi 100 ms

    def stop_plot(self):
        self.timer.stop()

    def clear_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.plot_widget.draw()
        self.plot_widget_1.draw()
        self.plot_widget_2.draw()
        self.plot_widget_3.draw()

    def update_plot(self):
        # Fake data (x,y,yaw)
        self.t += 0.1
        self.x_data.append(self.t/10)
        self.y_data.append(np.sin(self.t/5))
        self.yaw_data.append(np.sin(self.t/10)*30)

        # Vẽ lại
        self.ax1.plot(self.x_data, self.y_data, 'b-', label="Actual Path")
        self.plot_widget.draw()

        self.ax2.plot(self.t, self.x_data[-1] - (self.x_ref[0] if self.x_ref else 0), 'g.')
        self.plot_widget_1.draw()

        self.ax3.plot(self.t, self.y_data[-1] - (self.y_ref[0] if self.y_ref else 0), 'm.')
        self.plot_widget_2.draw()

        self.ax4.plot(self.t, self.yaw_data[-1], 'c.')
        self.plot_widget_3.draw()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
