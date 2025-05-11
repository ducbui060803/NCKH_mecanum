#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

// Declare for IMU 
Adafruit_BNO055 bno = Adafruit_BNO055(55);


const float encoder_resolution = 240.0; // Số xung trên một vòng quay

// How many motors
#define NMOTORS 4
volatile long encoderCount[NMOTORS] = {0};
unsigned long lastEncoderTime[NMOTORS] = {0};
// Pins
const int enca[NMOTORS] = {18,19,3,2};
const int encb[NMOTORS] = {31,38,49,A1};
const int pwm[] = {12,8,9,5};
const int in1[] = {34,37,43,A4};
const int in2[] = {35,36,42,A5};

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

double Setpoint[NMOTORS], SpeedMeasured[NMOTORS], PWM_Output[NMOTORS];

#define SPEED 100  // Tốc độ động cơ (0 - 255)

void setup() {
    Serial.begin(115200);
    // ========================== Init for motors =============================
    for(int k = 0; k < NMOTORS; k++)
    {
      pinMode(enca[k],INPUT);
      pinMode(encb[k],INPUT);
      pinMode(pwm[k],OUTPUT);
      pinMode(in1[k],OUTPUT);
      pinMode(in2[k],OUTPUT);
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

// Hàm điều khiển động cơ
void driveMotor(int inA, int inB, int pwm, int speed, bool forward) {
    digitalWrite(inA, forward ? HIGH : LOW);
    digitalWrite(inB, forward ? LOW : HIGH);
    analogWrite(pwm, speed);
}

// Hàm điều khiển toàn bộ robot
void move(int speed1, int speed2, int speed3, int speed4) {
      driveMotor(in1[0], in2[0], pwm[0], abs(speed1), speed1 > 0);
      driveMotor(in1[1], in2[1], pwm[1], abs(speed2), speed2 > 0);
      driveMotor(in1[2], in2[2], pwm[2], abs(speed3), speed3 > 0);
      driveMotor(in1[3], in2[3], pwm[3], abs(speed4), speed4 > 0);
}

// Hàm dừng robot
void stopRobot() {
    move(0, 0, 0, 0);
}

double readSpeed(int idx) {
    unsigned long now = millis();
    double dt = (now - lastEncoderTime[idx]) / 1000.0; // Đổi thành giây
    if (dt == 0) return 0;

    double speed = (encoderCount[idx] / encoder_resolution) / dt; // RPS (vòng/giây)
    encoderCount[idx] = 0;  // Reset bộ đếm
    lastEncoderTime[idx] = now;
    
    return speed*60 ; // Đổi sang RPM (vòng/phút)
}

void loop() {
    // Serial.print("Speed: ");
    for (int i = 0; i < NMOTORS; i++) {
        SpeedMeasured[i] = readSpeed(i);
        
        // Serial.print(SpeedMeasured[i]);
        // Serial.print(" ");
    } 

    Serial.print("\n");

    if (Serial.available()) {
        Serial.println("command");
        char command = Serial.read();
        Serial.println(command);
        
        switch (command) {
            case 'W': case 'w':  // Tiến
                move(-SPEED, SPEED, -SPEED, SPEED);
                Serial.println("hello");
                break;
            case 'S': case 's':  // Lùi
                move(SPEED, -SPEED, SPEED, -SPEED);
                break;
            case 'A': case 'a':  // Sang trái
                move(SPEED, SPEED, -SPEED, -SPEED);
                break;
            case 'D': case 'd':  // Sang phải
                move(-SPEED, -SPEED, SPEED, SPEED);
                break;
            case 'Q': case 'q':  // Quay trái
                move(SPEED, SPEED, SPEED, SPEED);
                break;
            case 'E': case 'e':  // Quay phải
                move(-SPEED, -SPEED, -SPEED, -SPEED);
                break;
            case 'X': case 'x':  // Dừng
                stopRobot();
                break;
            default:
                break;
        }
    
        
        // =============================== IMU Loop ==================================
        imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
        // Serial.print("Yaw: ");
        // Serial.print(euler.x());
        // Serial.print(" | Pitch: ");
        // Serial.print(euler.y());
        // Serial.print(" | Roll: ");
        // Serial.println(euler.z());
        delay(100);
    }
}