#!/usr/bin/env python3

#import rospy
#from geometry_msgs.msg import Vector3
import numpy as np
import math
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R

# Import the necessary libraries for socket communication
import socket
import time

updated_id11_once = False
updated_id4_once = False
def dis_filter(current_value, pre_ar, a):
    if (a == 1):
        thr = 3
    else:
        thr = 0.03
        
    # Nếu không có giá trị trước đó, lưu lại giá trị hiện tại và trả về True
    if (pre_ar == 0):
        pre_ar = current_value
        return True, pre_ar

    # Kiểm tra sự thay đổi giữa giá trị hiện tại và giá trị trước đó
    if abs(current_value - pre_ar) < thr:
        pre_ar = current_value  # Cập nhật giá trị trước đó
        return True, pre_ar  # Sự thay đổi đủ lớn, thực hiện hành động

    # Nếu sự thay đổi quá nhỏ, bỏ qua hành động
    return False, pre_ar

# Define the IP address and port number of the virtual machine
IPC_IP = "172.18.223.255"  # Thay bằng IP của IPC
PORT = 5005  

def send_pose(x, y, phi):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        pose_str = f"{x},{y},{phi}"
        sock.sendto(pose_str.encode(), (IPC_IP, PORT))
        print(f"Sent: {pose_str}")

# Load camera calibration data (intrinsic matrix and distortion coefficients)
camera_calibration_parameters_filename = 'calibration_chessboard_webcam.yaml'
# Initialize ROS node
#rospy.init_node('pose_estimation_publisher', anonymous=True)

# ROS publisher for the /odom topic
#pose_pub = rospy.Publisher('/odom', Vector3, queue_size=10)

def publish_pose(pose):
#    position = Vector3()
    position.x = pose['x']
    position.y = pose['y']
    position.z = pose['phi']
#    pose_pub.publish(position)

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def get_aruco(x0,y0,phi0,rvect,tvect):
    phi1 = math.atan(-rvect[2][0]/math.sqrt(math.pow(rvect[2][1],2)+math.pow(rvect[2][2],2)))
    d = math.sqrt(math.pow(tvect[0],2)+math.pow(tvect[2]+0.11,2))
    phiaruco = phi1+phi0
    phi2 = math.atan(tvect[0]/(tvect[2]+0.11))
    phi3 = phiaruco-phi2
    xaruco = x0 - d*math.cos(phi3)
    yaruco = y0 - d*math.sin(phi3)
    return xaruco,yaruco,phiaruco,d

def find_min_index(numbers):
    min_num = numbers[0]
    min_index = 0
    for i, num in enumerate(numbers):
        if num < min_num:
            min_num = num
            min_index = i
    return min_index

# You need to have your camera calibrated beforehand to get these values
cv_file = cv2.FileStorage(camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode('K').mat()
dist_coeffs = cv_file.getNode('D').mat()



# Initialize video capture with the IP camera URL
# cap = cv2.VideoCapture("rtsp://baoduc:baoduc@192.168.100.110:554/stream")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream from IP camera")
    exit()

# Initialize the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Initialize parameters for ArUco detection
parameters = aruco.DetectorParameters()

marker_size = 0.13  # Define the size of the marker in meters

# Read frames from the IP camera and process them
# Capsule 
# id_7 = [1.2, 0.4, - math.pi/2 ]
# id_6 = [1.4, 0.8, - math.pi/4 ]
# id_5 = [1.4,   1.2,    0]
# id_4 = [1.2,     1.8,      0]
# id_3 = [0.8,   2.4,  0]
# id_2 = [0.2,     2.5,  math.pi/2]
# id_1 = [-0.4,  2.5,  math.pi/2]
# id_0 = [-1,    2,    math.pi/2]
id_12 = [-1,  -1,  -math.pi]
id_11 = [-0.4,    -1.4,    -math.pi]
id_10 = [0.4, -1.4, - math.pi/2 ]
id_9 = [1, -1, - math.pi/2 ]
id_8 = [1.4,   -0.4,    -math.pi/2]
id_7 = [1.4,     0.4,      0]
id_6 = [1,   1,  0]
id_5 = [0.4,     1.4, 0]
id_4 = [-0.4,  1.4,  math.pi/2]
id_3 = [-1,    1,    math.pi/2]
id_2 = [-1.4,   0.4,  math.pi/2]
id_1 = [-1.4,     -0.4,  math.pi]
id_0 = [-1,  -1,  math.pi]

# marker_start = [id_0,id_1,id_2, id_3,id_4,id_5, id_6, id_7]
marker_start = [id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_7, id_8, id_9, id_10, id_11, id_12]
# marker_start = [id_0,id_1,id_2,id_3,id_4,id_5,id_6]
flag = False
height = 480
width = 640

# Set the loop rate (e.g., 10 Hz)
#rate = rospy.Rate(200)
marker_final = [-1, 2, math.pi/2]
threshold = 0.1
recent_pose = {'x': None, 'y': None, 'phi': None}

pre_x = 0
pre_y = -1
pre_angle = 180
pose_buffer = []

alpha = 0.5
a_a = 0.5
filtered_x = None
filtered_y = None
filtered_angle = None
def LowPassFilter(x, y, angle):
    global filtered_x, filtered_y, filtered_angle,alpha, a_a
    if filtered_x is None or filtered_y is None or filtered_angle is None:
        filtered_x = x
        filtered_y = y
        filtered_angle = angle
    else:
        filtered_x = alpha * x + (1 - alpha) * filtered_x
        filtered_y = alpha * y + (1 - alpha) * filtered_y
        filtered_angle = a_a * angle + (1 - a_a) * filtered_angle
    return filtered_x, filtered_y, filtered_angle
def should_filter_angle(vx, vy):
    # Nếu robot đang đi gần như thẳng, giữ filter góc
    # Nếu robot đi cong (vx thay đổi mạnh), bỏ filter
    speed = math.sqrt(vx ** 2 + vy ** 2)
    return speed > 0.1  # threshold tùy chỉnh

def smooth_pose(x, y, phi, window=5):
    global pose_buffer
    pose_buffer.append([x, y, phi])
    if len(pose_buffer) > window:
        pose_buffer.pop(0)
    arr = np.array(pose_buffer)
    smoothed = np.mean(arr, axis=0)
    
    # Nếu thay đổi quá nhỏ, giữ nguyên vị trí cũ để tránh "nhảy"
    if len(pose_buffer) >= 2:
        delta = np.linalg.norm(arr[-1, :2] - arr[-2, :2])
        if delta < 0.015:
            smoothed[:2] = arr[-2, :2]
    
    return smoothed[0], smoothed[1], smoothed[2]

def marker_area(corner):
    pts = corner[0]
    return cv2.contourArea(pts.astype(np.float32))

def normalize_angle_deg(angle):
    """Normalize angle to be within [-180, 180] and convert -180 to 180."""
    angle = (angle + 180) % 360 - 180
    if angle == -180:
        angle = 180
    return angle

while True:
#while not rospy.is_shutdown():
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    #frame = cv2.flip(frame,1)
    #height, width, m = frame.shape
    if not ret:
        print("Error: Could not read frame from IP camera")
        break

    # Detect ArUco markers in the image
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)
    filtered_corners = []
    filtered_ids = []
    area_threshold = 1000  # Thử nghiệm giá trị này
    # If markers are detected, estimate their pose
    if ids is not None:
        for i, corner in enumerate(corners):
            area = marker_area(corner)
            if area > area_threshold:
                filtered_corners.append(corner)
                filtered_ids.append(ids[i])

        if filtered_ids:
            flag = False

            all_marker = []
            distance_infor = []
            rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
            # Draw axes for each marker
            #print(rvecs)

            for i in range(len(ids)):
                cv2.aruco.drawDetectedMarkers(frame, corners)

                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size)
                #print(corners)
                # Extract translation (x, y, z)
                # Store the translation (i.e. position) information
                transform_translation_x = tvecs[i][0]
                transform_translation_y = tvecs[i][1]
                transform_translation_z = tvecs[i][2]
                t = [transform_translation_x,transform_translation_y,transform_translation_z]
                # Store the rotation information
                rotation_matrix,_ = cv2.Rodrigues(rvecs[i])
                #print(rotation_matrix)


                id = ids[i][0]
                if id <= 13:

                    x0 = marker_start[id][0]
                    y0 = marker_start[id][1]
                    phi0 = marker_start[id][2]

                    xaruco,yaruco,phiaruco, distance = get_aruco(x0,y0,phi0,rotation_matrix,t)
                    phiaruco = math.degrees(phiaruco)
                    
                    phiaruco = normalize_angle_deg(phiaruco)  # Ensure it is in [-180, 180]
                    if (id == 0 or id ==1) and not updated_id11_once:
                        if (phiaruco < 0) :
                            phiaruco = 180
                    if ((id == 11) or (id == 0)) and updated_id11_once:
                        if phiaruco > 0:
                            phiaruco = -180
                        updated_id11_once = True
                    print(f"kkk:{phiaruco}")
                    marker_info = [xaruco,yaruco,phiaruco]


                    all_marker.append(marker_info)
                    distance_infor.append(distance)
                else:
                    print('Wrong detect')
                    marker_final[0] = pre_x 
                    marker_final[1] = pre_y 
                    marker_final[2] = pre_angle 
                    flag = True

            if flag == False:
                print(distance_infor)
                # min_index = find_min_index(distance_infor)
                # marker_final = all_marker[min_index]
                weights = 1 / (np.array(distance_infor) + 1e-6)
                weights /= np.sum(weights)

                x_est = np.sum(weights * np.array([p[0] for p in all_marker]))
                y_est = np.sum(weights * np.array([p[1] for p in all_marker]))
                phi_est = np.sum(weights * np.array([p[2] for p in all_marker]))
                marker_final = [x_est, y_est, phi_est]
                # near_id = ids[min_index]

                x_raw = x_est
                y_raw = y_est
                phi_raw = phi_est
                # Sau đoạn này:
                marker_final = [x_est, y_est, phi_est]

                # Thêm LowPassFilter tại đây
                x_filt, y_filt, phi_filt = LowPassFilter(x_est, y_est, phi_est)
                # x_filt, y_filt, phi_filt = x_est, y_est, phi_est
              
                # Smooth toàn phần
                x_smooth, y_smooth, phi_smooth = smooth_pose(x_filt, y_filt, phi_filt)
                # x_smooth, y_smooth, phi_smooth = x_filt, y_filt, phi_filt

                # Điều kiện lọc
                flag_x, pre_x = dis_filter(x_smooth, pre_x, 0)
                flag_y, pre_y = dis_filter(y_smooth, pre_y, 0)
                flag_a, pre_angle = dis_filter(phi_smooth, pre_angle, 1)

                if not (flag_x or flag_y or flag_a):
                    # Không thay đổi gì
                    x_smooth = pre_x
                    y_smooth = pre_y
                    phi_smooth = pre_angle
                else:
                    if not should_filter_angle(x_smooth - pre_x, y_smooth - pre_y):
                        # Cho phép thay đổi góc nhanh (đường cong)
                        phi_smooth = phi_raw  # bỏ smoothing nếu muốn

                marker_final = [x_smooth, y_smooth, phi_smooth]
                pre_x = marker_final[0]
                pre_y = marker_final[1]
                pre_angle = marker_final[2]
                cv2.putText(frame, f"Position (x, y): ({marker_final[0]:.7f}, {marker_final[1]:.7f})",
                            (10, 30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Yaw: {marker_final[2]:.7f} degrees", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
                # cv2.putText(frame, f"id: {near_id}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #             (0, 0, 255), 2) 
                print(f"{marker_final[0],marker_final[1]}")

                if recent_pose['x'] is not None and abs(
                    marker_final[0] - recent_pose['x']) > threshold: transform_translation_x = recent_pose['x']

                if recent_pose['y'] is not None and abs(
                    marker_final[1] - recent_pose['y']) > threshold: transform_translation_y = recent_pose['y']


                recent_pose['x'] = marker_final[0]
                recent_pose['y'] = marker_final[1]
                recent_pose['phi'] = marker_final[2]
                # update_trajectory()
                # Store values in the pose dictionary
            pose = {}
            pose['x'] = marker_final[0]
            # pose['x'] = 0
            pose['y'] = marker_final[1]
            #pose['y'] = y_d
            pose['phi'] = marker_final[2]*math.pi/180
            # pose['phi'] = math.pi/2
            position = (float(pose['x']), float(pose['y']), float(pose['phi']))

            # Publish the position and yaw angle
#            publish_pose(pose)

            # Send the position and yaw angle to the virtual machine
            send_pose(float(pose['x']), float(pose['y']), float(pose['phi']))
            time.sleep(0.01)

            # Print the position and yaw angle
            # print(f"Position (x, y): ({transform_translation_x}, {transform_translation_y})")
            # print(f"Yaw angle: {yaw_z_deg} degrees")
#            rospy.loginfo(pose)

            # Display the image with drawn axes
            # Draw coordinate axes (red for X, green for Y)
            # Draw a red vertical line through the center of the image
            cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)  # Red vertical line
            # Draw a green horizontal line through the center of the image
            # cv_file
            cv2.line(frame, (0, height // 2), (width, height // 2), (0, 0, 255), 2)  # Green horizontal line
            #fr = cv2.resize(frame,(3000,3000))
    else:
        marker_final[0] = pre_x 
        marker_final[1] = pre_y 
        marker_final[2] = pre_angle 
        pose = {}
        pose['x'] = marker_final[0]
        # pose['x'] = 0
        pose['y'] = marker_final[1]
        #pose['y'] = y_d
        pose['phi'] = marker_final[2]*math.pi/180
        # pose['phi'] = math.pi/2
        position = (float(pose['x']), float(pose['y']), float(pose['phi']))

        # Publish the position and yaw angle
#            publish_pose(pose)

        # Send the position and yaw angle to the virtual machine
        send_pose(float(pose['x']), float(pose['y']), float(pose['phi']))
        time.sleep(0.01)
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Sleep to maintain the loop rate
#    rate.sleep()
# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
