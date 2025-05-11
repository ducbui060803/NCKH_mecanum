#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

// Declare for IMU 
Adafruit_BNO055 bno = Adafruit_BNO055(55);
#define MAX_SPEED 100
#define MAX 48
// Số lượng động cơ
#define NMOTORS 4
#define PI 3.14159
// Kích thước robot (m)
#define L1 0.11  // Khoảng cách từ trục OpYp đến bánh xe
#define L2 0.1 // Khoảng cách từ trục OpXp đến bánh xe
#define r 0.035   // Bán kính bánh xe

bool started = false;
// Thông số PID

double Kp = 2.0, Ki = 0.9, Kd = 0.01; 
double Vx, Vy, V, yaw_deg, yaw, theta_dot, pre_yaw = 90;
double vx = 0, vy = 0, omega = 0;

double prevError[NMOTORS] = {0};
double integral[NMOTORS] = {0};

// Encoder và động cơ

const int enca[NMOTORS] = {18, 19, 3, 2};  // Encoder A
const int encb[NMOTORS] = {31, 38, 49, A1}; // Encoder B
const int pwm[NMOTORS]  = {12, 8, 9, 5};
const int in1[NMOTORS]  = {34, 37, 43, A4};
const int in2[NMOTORS]  = {35, 36, 42, A5};

// Giá trị đặt và giá trị đo
double Setpoint[NMOTORS], SpeedMeasured[NMOTORS], PWM_Output[NMOTORS];

// Biến cho encoder
volatile long encoderCount[NMOTORS] = {0};
unsigned long lastEncoderTime[NMOTORS] = {0};
const float encoder_resolution = 240.0; // Số xung trên một vòng quay

// Hàm ngắt để đếm xung encoder
void encoderISR0() { 
    int b = digitalRead(encb[0]);
    if(b > 0){
        encoderCount[0]++;
    }
    else{
        encoderCount[0]--;
    }

    }
void encoderISR1() { 
    int b = digitalRead(encb[1]);
    if(b > 0){
        encoderCount[1]++;
    }
    else{
        encoderCount[1]--;
    }
    }
void encoderISR2() { 
    int b = digitalRead(encb[2]);
    if(b > 0){
        encoderCount[2]++;
    }
    else{
        encoderCount[2]--;
    }
    }
void encoderISR3() { 
    int b = digitalRead(encb[3]);
    if(b > 0){
        encoderCount[3]++;
    }
    else{
        encoderCount[3]--;
    }
    }

void setup() {
    Serial3.begin(115200); // UART giao tiếp với ROS
    Serial.begin(115200);    // Debug
    

    for (int i = 0; i < NMOTORS; i++) {
        Setpoint[i] = 0;
        encoderCount[i] = 0;
        lastEncoderTime[i] = 0;
        prevError[i] = millis();
        integral[i] = 0;
        driveMotor(i,0);
        pinMode(enca[i], INPUT);
        pinMode(encb[i], INPUT);
        pinMode(pwm[i], OUTPUT);
        pinMode(in1[i], OUTPUT);
        pinMode(in2[i], OUTPUT);
    }

    // Gán ngắt encoder
    attachInterrupt(digitalPinToInterrupt(enca[0]), encoderISR0, RISING);
    attachInterrupt(digitalPinToInterrupt(enca[1]), encoderISR1, RISING);
    attachInterrupt(digitalPinToInterrupt(enca[2]), encoderISR2, RISING);
    attachInterrupt(digitalPinToInterrupt(enca[3]), encoderISR3, RISING);

    // =========================== Init for IMU ================================
    if (!bno.begin()) {
    Serial.println("Không tìm thấy BNO055. Kiểm tra dây nối!");
    while (1);
    }
    Serial.println("Đã kết nối BNO055 thành công!");
}

// Hàm đọc tốc độ từ encoder
double readSpeed(int idx) {
    unsigned long now = millis();
    double dt = (now - lastEncoderTime[idx]) / 1000.0; // Đổi thành giây
    if (dt == 0) return 0;

    double speed = (encoderCount[idx] / encoder_resolution) / dt; // RPS (vòng/giây)
    encoderCount[idx] = 0;  // Reset bộ đếm
    lastEncoderTime[idx] = now;
    
    return speed *60.0; // Đổi sang RPM (vòng/phút)
}

// PID Controller
double computePID(int idx, double setpoint, double measured, double dt) {
    // if (setpoint * measured < 0) {
    //     measured = -measured;  // Flip measured if it's in the opposite direction of setpoint
    // }
    double error = setpoint - measured;
    integral[idx] += error * dt;
    double derivative = (error - prevError[idx]) / dt;
    prevError[idx] = error;

    double output = Kp * error + Ki * integral[idx] + Kd * derivative;
    
    //return constrain(measured + output, -MAX_SPEED, MAX_SPEED); // Giới hạn PWM
    return constrain(setpoint, -MAX_SPEED, MAX_SPEED); // Giới hạn PWM
}

// Điều khiển động cơ
void driveMotor(int idx, double speed) {
    bool forward = (speed > 0);
    speed = abs(speed);

    digitalWrite(in1[idx], forward ? HIGH : LOW);
    digitalWrite(in2[idx], forward ? LOW : HIGH);

    // Serial.print(min(speed, 255));
    // Serial.print('\n');
    analogWrite(pwm[idx], min(speed, 255));
}

// Điều khiển toàn bộ robot
void move(double vx, double vy, double omega) {
    double v1 = (vx - vy - (L1 + L2) * omega) / r;
    double v2 = (vx + vy + (L1 + L2) * omega) / r;
    double v3 = (vx - vy + (L1 + L2) * omega) / r;
    double v4 = (vx + vy - (L1 + L2) * omega) / r;

    double max_speed = max(max(abs(v1), abs(v2)), max(abs(v3), abs(v4)));
    if (max_speed > 1.0) {  // Chuẩn hóa nếu cần
        v1 /= max_speed;
        v2 /= max_speed;
        v3 /= max_speed;
        v4 /= max_speed;
    }

    Setpoint[0] = v1 * MAX;
    Setpoint[1] = v2 * MAX;
    Setpoint[2] = v3 * MAX;
    Setpoint[3] = v4 * MAX;

    Serial.print("SET POINT  ");
    Serial.print(Setpoint[0]);
    Serial.print(" ");
    Serial.print(Setpoint[1]);
    Serial.print(" ");
    Serial.print(Setpoint[2]);
    Serial.print(" ");
    Serial.print(Setpoint[3]);
    Serial.print("\n");
}

// Hàm dừng robot
void stopRobot() {
    for (int i = 0; i < NMOTORS; i++) {
        analogWrite(pwm[i], 0);
    }
}

void loop() {
    static unsigned long lastTime = millis();
    unsigned long now = millis();
    double dt = (now - lastTime) / 1000.0; // Chuyển đổi thành giây
    
    lastTime = now;
    if('S' == Serial.read()) {
        stopRobot();
    }
    // ==================================== UART Control ======================================
    if (Serial3.available()) {
        String data = Serial3.readStringUntil('\n');
        int firstSpaceIndex = data.indexOf(' ');
        int secondSpaceIndex = data.indexOf(' ', firstSpaceIndex + 1);

        String vxStr = data.substring(0, firstSpaceIndex);
        String vyStr = data.substring(firstSpaceIndex + 1, secondSpaceIndex);
        String omegaStr = data.substring(secondSpaceIndex + 1);

        vx = vxStr.toDouble();
        vy = vyStr.toDouble();
        omega = omegaStr.toDouble();

        /*Serial.print(vx);
        Serial.print(" ");
        Serial.print(vy);
        Serial.print(" ");
        Serial.print(omega);
        Serial.print("\n");*/
        started = true;
        move(vx, vy, omega);

    }

    // ====================================== IMU Loop ========================================
    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
    imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

    
    yaw_deg = euler.x(); 
    if (yaw_deg > 180){
        yaw_deg = yaw_deg - 360;
    }  // deg/s
    yaw_deg = yaw_deg + 180;
    yaw = yaw_deg * (PI / 180.0);  // rad/s
    // yaw = 0;    


    theta_dot = gyro.z();
    theta_dot = theta_dot * (PI / 180.0);  // rad/s

    Serial.print("Theta_dot: ");
    Serial.print(theta_dot);

    Serial.print("\n");
    // ==================================== Encoder Read ======================================
    if (started == true) {
        for (int i = 0; i < NMOTORS; i++) {
            SpeedMeasured[i] = readSpeed(i);    
            PWM_Output[i] = computePID(i, Setpoint[i], SpeedMeasured[i], dt);
            driveMotor(i, PWM_Output[i]);
        }
    }

    Serial.print("Encoder  ");
    Serial.print(SpeedMeasured[0]);
    Serial.print(" ");
    Serial.print(SpeedMeasured[1]);
    Serial.print(" ");
    Serial.print(SpeedMeasured[2]);
    Serial.print(" ");
    Serial.print(SpeedMeasured[3]);
    Serial.print("\n");
    Serial.print("Drive motor  ");
    Serial.print(PWM_Output[0]);
    Serial.print(" ");
    Serial.print(PWM_Output[1]);
    Serial.print(" ");
    Serial.print(PWM_Output[2]);
    Serial.print(" ");
    Serial.print(PWM_Output[3]);
    Serial.print("\n");
    // ======================= Calculate Robot Speed from wheel speed =========================
    Vx = (SpeedMeasured[0] + SpeedMeasured[1] + SpeedMeasured[2] + SpeedMeasured[3]) * r / 4.0;
    Vy = (-SpeedMeasured[0] + SpeedMeasured[1] + SpeedMeasured[2] - SpeedMeasured[3]) * r / 4.0;
    // theta_dot =  (-SpeedMeasured[0] + SpeedMeasured[1] - SpeedMeasured[2] + SpeedMeasured[3]) * r / (4.0 * (L1 + L2));
    // toán vận tốc tổng hợp của xe (vận tốc theo hướng tổng hợp)
    V = sqrt(Vx * Vx + Vy * Vy);

    // Truyền giá trị V qua UART (Serial1)
    Serial3.print(V, 3);       // Truyền V với 3 chữ số thập phân
    Serial3.print(" ");
    Serial3.print(yaw, 3);     // Truyen yaw với 3 chữ số thập phân
    Serial3.print(" ");
    Serial3.println(theta_dot, 3); // Truyền theta_dot với 3 chữ số thập phân

    delay(100); // Thêm một chút thời gian trễ để ổn định
}
