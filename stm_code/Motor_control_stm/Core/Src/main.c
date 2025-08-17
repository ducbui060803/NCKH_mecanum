/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define PWM_FAST 					600
#define PWM_SLOW 					300
#define PPR 	 					240        // Pulses per revolution
#define DT 		 					0.01f      // 10ms (chu kỳ gọi hàm đo)
#define BNO055_ADDRESS 				0x28 << 1  // Shifted 7-bit to 8-bit I2C address
#define BNO055_EULER_H_LSB 			0x1A
#define BNO055_GYRO_DATA_Z_LSB 		0x1E
#define RX_BUFFER_SIZE 				100
#define TX_BUFFER_SIZE 				100
// Kích thước robot (m)
#define L1 							0.11  	// Khoảng cách từ trục OpYp đến bánh xe
#define L2 							0.1 	// Khoảng cách từ trục OpXp đến bánh xe
#define r 							0.035   // Bán kính bánh xe
#define MAX_SPEED 					100
#define PWM_MAX 					300
#define MANUAL_TIMEOUT 				3000  // 3s timeout
#define TRUE						1
#define FALSE						0
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c3;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim5;
TIM_HandleTypeDef htim9;
TIM_HandleTypeDef htim10;
TIM_HandleTypeDef htim11;

UART_HandleTypeDef huart6;
DMA_HandleTypeDef hdma_usart6_rx;
DMA_HandleTypeDef hdma_usart6_tx;

/* USER CODE BEGIN PV */

// GPIO chân điều khiển chiều quay (IN1 & IN2)
//GPIO_TypeDef* in1_port[4] = { GPIOA, GPIOD, GPIOB, GPIOC };
//uint16_t in1_pin[4]       = { GPIO_PIN_15, GPIO_PIN_2, GPIO_PIN_9, GPIO_PIN_2 };
//
//GPIO_TypeDef* in2_port[4] = { GPIOC, GPIOD, GPIOE, GPIOC };
//uint16_t in2_pin[4]       = { GPIO_PIN_11, GPIO_PIN_6, GPIO_PIN_3, GPIO_PIN_0 };
GPIO_TypeDef* in1_port[4] = { GPIOD, GPIOA, GPIOC, GPIOE };
uint16_t in1_pin[4]       = { GPIO_PIN_6, GPIO_PIN_15, GPIO_PIN_2, GPIO_PIN_3 };

GPIO_TypeDef* in2_port[4] = { GPIOD, GPIOC, GPIOC, GPIOB };
uint16_t in2_pin[4]       = { GPIO_PIN_2, GPIO_PIN_11, GPIO_PIN_0, GPIO_PIN_9 };
// PWM kênh và timer
TIM_HandleTypeDef* htim_pwm[4] = { &htim5, &htim9, &htim10, &htim5 };
uint32_t tim_channel[4]        = { TIM_CHANNEL_3, TIM_CHANNEL_2, TIM_CHANNEL_1, TIM_CHANNEL_4 };
UART_HandleTypeDef huart6;
/* NOTE: Swap M1 vs M2*/
volatile uint8_t encoder_flag =  FALSE;
volatile uint8_t imu_flag = FALSE;
char rx_data;                  		// ký tự nhận được tạm thời
char temp_line[100];
uint8_t rx_buffer[RX_BUFFER_SIZE];  // Vòng đệm DMA nhận
char tx_buffer[TX_BUFFER_SIZE];  // Buffer gửi
uint8_t rx_index 		= 0;
uint8_t started 		= 0;
uint8_t uart_tx_ready 	= 0;
uint8_t temp_line_index = 0;
uint8_t send_flag 		= 0;
uint8_t manual_mode = 0;
uint32_t last_manual_time = 0;
int16_t encoder_current[4];
int16_t encoder_past[4];
int16_t delta_encoder[4];
int16_t speed_rpm[4];
float yaw_deg 			= 0;
float yaw 				= 0;
float theta_dot 		= 0;
float vx 				= 0;
float vy  				= 0;
float omega 			= 0;
float V 				= 0;
float V_send 			= 0;
float yaw_send 			= 0;
float theta_dot_send 	= 0;

float Setpoint[4];
float SpeedMeasured[4];
float PWM_Output[4];
float integral[4] = {0};
float prevError[4] = {0};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_I2C3_Init(void);
static void MX_USART6_UART_Init(void);
static void MX_TIM5_Init(void);
static void MX_TIM9_Init(void);
static void MX_TIM10_Init(void);
static void MX_TIM11_Init(void);
/* USER CODE BEGIN PFP */
void user_init(void);
void update_encoder_speed(void);
void read_IMU(void);
void parse_uart_line(char *line);
void move(float vx, float vy, float omega);
void driveMotor(int idx, float speed);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* Timer callback */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM11)          // kiểm tra đúng timer
    {
        // Đặt code cần chạy mỗi chu kỳ ở đây
    	encoder_flag = TRUE;
    	imu_flag = TRUE;
    }
}

/* UART callback */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART6)
    {
        uart_tx_ready = 1;
    }
}

void user_init()
{
	HAL_TIM_PWM_Start(&htim5, TIM_CHANNEL_3); //M2
	HAL_TIM_PWM_Start(&htim5, TIM_CHANNEL_4); //M3
	HAL_TIM_PWM_Start(&htim9, TIM_CHANNEL_2); //M1
	HAL_TIM_PWM_Start(&htim10, TIM_CHANNEL_1); //M4

	HAL_TIM_Encoder_Start(&htim1, TIM_CHANNEL_ALL); //M1
	HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL); //M3
	HAL_TIM_Encoder_Start(&htim3, TIM_CHANNEL_ALL); //M4
	HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL); //M2

	__HAL_TIM_SET_COMPARE(&htim5, TIM_CHANNEL_3, 0);
	__HAL_TIM_SET_COMPARE(&htim5, TIM_CHANNEL_4, 0);
	__HAL_TIM_SET_COMPARE(&htim9, TIM_CHANNEL_2, 0);
	__HAL_TIM_SET_COMPARE(&htim10, TIM_CHANNEL_1, 0);
	HAL_UART_Receive_DMA(&huart6, rx_buffer, RX_BUFFER_SIZE);
	HAL_TIM_Base_Start_IT(&htim11);
	// Set chiều quay THUẬN CHIỀU KIM ĐỒNG HỒ:
	// M1: IN1_1 = 1, IN2_1 = 0
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_15, GPIO_PIN_SET);   // IN1_1
	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_11, GPIO_PIN_RESET); // IN2_1

	// M2: IN1_2 = 1, IN2_2 = 0
	HAL_GPIO_WritePin(GPIOD, GPIO_PIN_2, GPIO_PIN_RESET);    // IN1_2
	HAL_GPIO_WritePin(GPIOD, GPIO_PIN_6, GPIO_PIN_SET);  // IN2_2

	// M3: IN1_3 = 1, IN2_3 = 0
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_RESET);    // IN1_3
	HAL_GPIO_WritePin(GPIOE, GPIO_PIN_3, GPIO_PIN_SET);  // IN2_3

	// M4: IN1_4 = 1, IN2_4 = 0
	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_0, GPIO_PIN_SET);    // IN1_4
	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_2, GPIO_PIN_RESET);  // IN2_4

	 // Reset BNO055
	uint8_t reset_cmd = 0x20;
	HAL_I2C_Mem_Write(&hi2c3, BNO055_ADDRESS, 0x3F, 1, &reset_cmd, 1, HAL_MAX_DELAY);
	HAL_Delay(3000);  // Rất quan trọng!

	// Set to config mode
	uint8_t config_mode = 0x00;
	HAL_I2C_Mem_Write(&hi2c3, BNO055_ADDRESS, 0x3D, 1, &config_mode, 1, HAL_MAX_DELAY);
	HAL_Delay(25);

	// Set to NDOF mode
	uint8_t ndof_mode = 0x0C;
	HAL_I2C_Mem_Write(&hi2c3, BNO055_ADDRESS, 0x3D, 1, &ndof_mode, 1, HAL_MAX_DELAY);
	HAL_Delay(20);
}

void update_encoder_speed(void)
{
	for (int i = 0; i < 4; i++)
	{
		switch (i)
		{
			case 0:
				encoder_current[i] = __HAL_TIM_GET_COUNTER(&htim4); //M2
				break;
			case 1:
				encoder_current[i] = __HAL_TIM_GET_COUNTER(&htim1); //M1
				break;
			case 2:
				encoder_current[i] = __HAL_TIM_GET_COUNTER(&htim3); //M3
				break;
			case 3:
				encoder_current[i] = __HAL_TIM_GET_COUNTER(&htim2); //M4
				break;
			default:
				break;
		}

		delta_encoder[i] = encoder_current[i] - encoder_past[i];

	    // Nếu counter bị overflow, xử lý wrap-around
	    if (delta_encoder[i] > 32767)
		{
	    	delta_encoder[i] -= 65536;
		}

	    if (delta_encoder[i] < -32768)
		{
	    	delta_encoder[i] += 65536;
		}

	    encoder_past[i] = encoder_current[i];

	    float speed_rps = (float) delta_encoder[i] / PPR / DT;
	    speed_rpm[i] = speed_rps * 60.0f;
	}
}

void read_IMU(void)
{
    uint8_t buffer[2];

    // ==== Read Yaw (Euler Heading) ====
    HAL_I2C_Mem_Read(&hi2c3, BNO055_ADDRESS, BNO055_EULER_H_LSB, 1, buffer, 2, HAL_MAX_DELAY);
    int16_t yaw_raw = (int16_t)((buffer[1] << 8) | buffer[0]);
    yaw_deg = ((float)yaw_raw) / 16.0f;  // 1° = 16 LSB
    if (yaw_deg > 180)
        yaw_deg -= 360;
    yaw_deg += 180;
    yaw = yaw_deg * (M_PI / 180.0f);  // rad

    // ==== Read Gyro Z (yaw rate) ====
    HAL_I2C_Mem_Read(&hi2c3, BNO055_ADDRESS, BNO055_GYRO_DATA_Z_LSB, 1, buffer, 2, HAL_MAX_DELAY);
    int16_t gyro_z_raw = (int16_t)((buffer[1] << 8) | buffer[0]);
    theta_dot = ((float)gyro_z_raw) / 16.0f;  // deg/s
    theta_dot *= (M_PI / 180.0f);  // rad/s
}

void check_uart_command(void)
{
    static uint16_t old_pos = 0;
    uint16_t new_pos = RX_BUFFER_SIZE - __HAL_DMA_GET_COUNTER(huart6.hdmarx);  // vị trí mới

    if (new_pos != old_pos)
    {
        while (old_pos != new_pos)
        {
            char c = rx_buffer[old_pos++];

            // Xử lý ký tự c (dùng buffer đệm riêng)
            if (c == '\n')
            {
                // Kết thúc chuỗi, parse
                parse_uart_line(temp_line);
                temp_line_index = 0;
                memset(temp_line, 0, sizeof(temp_line));
            }
            else if (temp_line_index < sizeof(temp_line) - 1)
            {
                temp_line[temp_line_index++] = c;
            }

            if (old_pos >= RX_BUFFER_SIZE)
                old_pos = 0;
        }
    }
}

void parse_uart_line(char *line)
{
    float vx_local = 0, vy_local = 0, omega_local = 0;

    char *vxStr = strtok(line, " ");
    char *vyStr = strtok(NULL, " ");
    char *omegaStr = strtok(NULL, " ");

    if (vxStr && vyStr && omegaStr)
    {
        vx_local = atof(vxStr);
        vy_local = atof(vyStr);
        omega_local = atof(omegaStr);

        vx = vx_local;
        vy = vy_local;
        omega = omega_local;

        started = 1;
        move(vx, vy, omega);
    }

    manual_mode = 1;
    last_manual_time = HAL_GetTick();
}

void move(float vx, float vy, float omega)
{
    float v1 = (vx - vy - (L1 + L2) * omega) / r;
    float v2 = (vx + vy + (L1 + L2) * omega) / r;
    float v3 = (vx - vy + (L1 + L2) * omega) / r;
    float v4 = (vx + vy - (L1 + L2) * omega) / r;

    float max_speed = fmaxf(fmaxf(fabsf(v1), fabsf(v2)), fmaxf(fabsf(v3), fabsf(v4)));
    if (max_speed > 1.0f)
    {
        v1 /= max_speed;
        v2 /= max_speed;
        v3 /= max_speed;
        v4 /= max_speed;
    }

    Setpoint[0] = v1 * MAX_SPEED;
    Setpoint[1] = v2 * MAX_SPEED;
    Setpoint[2] = v3 * MAX_SPEED;
    Setpoint[3] = v4 * MAX_SPEED;

    driveMotor(0, Setpoint[0] / MAX_SPEED);  // Normalized từ -1.0 đến 1.0
    driveMotor(1, Setpoint[1] / MAX_SPEED);
    driveMotor(2, Setpoint[2] / MAX_SPEED);
    driveMotor(3, Setpoint[3] / MAX_SPEED);
}

void driveMotor(int idx, float speed)
{
    if (idx < 0 || idx >= 4) return;

    uint8_t forward = (speed >= 0);
    float abs_speed = fabsf(speed);

    if (abs_speed > 1.0f) abs_speed = 1.0f;  // Giới hạn từ -1.0 đến 1.0

    // Tính giá trị PWM theo PWM_MAX
    uint32_t pwm_value = (uint32_t)(abs_speed * PWM_MAX);

    // Điều khiển chiều
    HAL_GPIO_WritePin(in1_port[idx], in1_pin[idx], forward ? GPIO_PIN_SET : GPIO_PIN_RESET);
    HAL_GPIO_WritePin(in2_port[idx], in2_pin[idx], forward ? GPIO_PIN_RESET : GPIO_PIN_SET);

    // Xuất PWM
    __HAL_TIM_SET_COMPARE(htim_pwm[idx], tim_channel[idx], pwm_value);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */

	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_DMA_Init();
	MX_TIM1_Init();
	MX_TIM2_Init();
	MX_TIM3_Init();
	MX_TIM4_Init();
	MX_I2C3_Init();
	MX_USART6_UART_Init();
	MX_TIM5_Init();
	MX_TIM9_Init();
	MX_TIM10_Init();
	MX_TIM11_Init();
	/* USER CODE BEGIN 2 */
	user_init();
	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1)
	{
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
		if (TRUE == encoder_flag)
		{
			update_encoder_speed();
			encoder_flag = FALSE;
		}
		if (TRUE == imu_flag)
		{
			read_IMU();
			imu_flag = FALSE;
		}

		check_uart_command();

		if ((manual_mode) && (HAL_GetTick() - last_manual_time > MANUAL_TIMEOUT))
		{
			manual_mode = 0;
			move(0, 0, 0);  // Dừng robot nếu timeout
		}

		if (send_flag && uart_tx_ready)
		{
			send_flag = 0;
			uart_tx_ready = 0;

			snprintf(tx_buffer, sizeof(tx_buffer), "%.3f %.3f %.3f\n", V_send, yaw_send, theta_dot_send);
			HAL_UART_Transmit_DMA(&huart6, (uint8_t *)tx_buffer, strlen(tx_buffer));
		}

//		// Forward
//		driveMotor(0, 200);
//		driveMotor(1, 200);
//		driveMotor(2, 200);
//		driveMotor(3, 200);
//		HAL_Delay(3000);
//
//		// Stop
//		for (int i = 0; i < 4; i++)
//		{
//			driveMotor(i, 0);
//		}
//		HAL_Delay(3000);
//
//		// Backward
//		driveMotor(0, -200);
//		driveMotor(1, -200);
//		driveMotor(2, -200);
//		driveMotor(3, -200);
//		HAL_Delay(3000);
//
//		// Stop
//		for (int i = 0; i < 4; i++)
//		{
//			driveMotor(i, 0);
//		}
//		HAL_Delay(3000);
//
//		// Left
//		driveMotor(0, -200);
//		driveMotor(1, 200);
//		driveMotor(2, -200);
//		driveMotor(3, 200);
//		HAL_Delay(3000);
//
//		// Stop
//		for (int i = 0; i < 4; i++)
//		{
//			driveMotor(i, 0);
//		}
//		HAL_Delay(3000);
//
//		// Right
//		driveMotor(0, 200);
//		driveMotor(1, -200);
//		driveMotor(2, 200);
//		driveMotor(3, -200);
//		HAL_Delay(3000);
//
//		// Stop
//		for (int i = 0; i < 4; i++)
//		{
//			driveMotor(i, 0);
//		}
//		HAL_Delay(3000);
	}
	/* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 144;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 6;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C3_Init(void)
{

  /* USER CODE BEGIN I2C3_Init 0 */

  /* USER CODE END I2C3_Init 0 */

  /* USER CODE BEGIN I2C3_Init 1 */

  /* USER CODE END I2C3_Init 1 */
  hi2c3.Instance = I2C3;
  hi2c3.Init.ClockSpeed = 100000;
  hi2c3.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c3.Init.OwnAddress1 = 0;
  hi2c3.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c3.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c3.Init.OwnAddress2 = 0;
  hi2c3.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c3.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C3_Init 2 */

  /* USER CODE END I2C3_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 65535;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 65535;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_FALLING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim2, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 65535;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 65535;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim4, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM5_Init(void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 72;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 1000;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim5, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim5) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim5, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim5, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM5_Init 2 */

  /* USER CODE END TIM5_Init 2 */
  HAL_TIM_MspPostInit(&htim5);

}

/**
  * @brief TIM9 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM9_Init(void)
{

  /* USER CODE BEGIN TIM9_Init 0 */

  /* USER CODE END TIM9_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM9_Init 1 */

  /* USER CODE END TIM9_Init 1 */
  htim9.Instance = TIM9;
  htim9.Init.Prescaler = 72;
  htim9.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim9.Init.Period = 1000;
  htim9.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim9.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim9) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim9, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim9) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim9, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM9_Init 2 */

  /* USER CODE END TIM9_Init 2 */
  HAL_TIM_MspPostInit(&htim9);

}

/**
  * @brief TIM10 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM10_Init(void)
{

  /* USER CODE BEGIN TIM10_Init 0 */

  /* USER CODE END TIM10_Init 0 */

  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM10_Init 1 */

  /* USER CODE END TIM10_Init 1 */
  htim10.Instance = TIM10;
  htim10.Init.Prescaler = 72;
  htim10.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim10.Init.Period = 1000;
  htim10.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim10.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim10) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim10) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim10, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM10_Init 2 */

  /* USER CODE END TIM10_Init 2 */
  HAL_TIM_MspPostInit(&htim10);

}

/**
  * @brief TIM11 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM11_Init(void)
{

  /* USER CODE BEGIN TIM11_Init 0 */

  /* USER CODE END TIM11_Init 0 */

  /* USER CODE BEGIN TIM11_Init 1 */

  /* USER CODE END TIM11_Init 1 */
  htim11.Instance = TIM11;
  htim11.Init.Prescaler = 7200-1;
  htim11.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim11.Init.Period = 99;
  htim11.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim11.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim11) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM11_Init 2 */

  /* USER CODE END TIM11_Init 2 */

}

/**
  * @brief USART6 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART6_UART_Init(void)
{

  /* USER CODE BEGIN USART6_Init 0 */

  /* USER CODE END USART6_Init 0 */

  /* USER CODE BEGIN USART6_Init 1 */

  /* USER CODE END USART6_Init 1 */
  huart6.Instance = USART6;
  huart6.Init.BaudRate = 115200;
  huart6.Init.WordLength = UART_WORDLENGTH_8B;
  huart6.Init.StopBits = UART_STOPBITS_1;
  huart6.Init.Parity = UART_PARITY_NONE;
  huart6.Init.Mode = UART_MODE_TX_RX;
  huart6.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart6.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart6) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART6_Init 2 */

  /* USER CODE END USART6_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);
  /* DMA2_Stream6_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream6_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream6_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(IN2_3_GPIO_Port, IN2_3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, IN2_4_Pin|IN1_4_Pin|IN2_1_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, LD4_Pin|LD3_Pin|LD5_Pin|LD6_Pin
                          |IN1_2_Pin|IN2_2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(IN1_1_GPIO_Port, IN1_1_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(IN1_3_GPIO_Port, IN1_3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : DATA_Ready_Pin */
  GPIO_InitStruct.Pin = DATA_Ready_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(DATA_Ready_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : IN2_3_Pin */
  GPIO_InitStruct.Pin = IN2_3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(IN2_3_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : INT1_Pin INT2_Pin MEMS_INT2_Pin */
  GPIO_InitStruct.Pin = INT1_Pin|INT2_Pin|MEMS_INT2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pins : IN2_4_Pin IN1_4_Pin IN2_1_Pin */
  GPIO_InitStruct.Pin = IN2_4_Pin|IN1_4_Pin|IN2_1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : PA0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : LD4_Pin LD3_Pin LD5_Pin LD6_Pin
                           IN1_2_Pin IN2_2_Pin */
  GPIO_InitStruct.Pin = LD4_Pin|LD3_Pin|LD5_Pin|LD6_Pin
                          |IN1_2_Pin|IN2_2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pin : VBUS_FS_Pin */
  GPIO_InitStruct.Pin = VBUS_FS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(VBUS_FS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : OTG_FS_ID_Pin OTG_FS_DM_Pin OTG_FS_DP_Pin */
  GPIO_InitStruct.Pin = OTG_FS_ID_Pin|OTG_FS_DM_Pin|OTG_FS_DP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_FS;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : IN1_1_Pin */
  GPIO_InitStruct.Pin = IN1_1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(IN1_1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PD4 */
  GPIO_InitStruct.Pin = GPIO_PIN_4;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pin : OTG_FS_OverCurrent_Pin */
  GPIO_InitStruct.Pin = OTG_FS_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_FS_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PB5 */
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : IN1_3_Pin */
  GPIO_InitStruct.Pin = IN1_3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(IN1_3_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
