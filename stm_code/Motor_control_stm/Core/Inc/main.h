/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define DATA_Ready_Pin GPIO_PIN_2
#define DATA_Ready_GPIO_Port GPIOE
#define IN2_3_Pin GPIO_PIN_3
#define IN2_3_GPIO_Port GPIOE
#define INT1_Pin GPIO_PIN_4
#define INT1_GPIO_Port GPIOE
#define INT2_Pin GPIO_PIN_5
#define INT2_GPIO_Port GPIOE
#define PWM_1_Pin GPIO_PIN_6
#define PWM_1_GPIO_Port GPIOE
#define PC14_OSC32_IN_Pin GPIO_PIN_14
#define PC14_OSC32_IN_GPIO_Port GPIOC
#define PC15_OSC32_OUT_Pin GPIO_PIN_15
#define PC15_OSC32_OUT_GPIO_Port GPIOC
#define PH0_OSC_IN_Pin GPIO_PIN_0
#define PH0_OSC_IN_GPIO_Port GPIOH
#define PH1_OSC_OUT_Pin GPIO_PIN_1
#define PH1_OSC_OUT_GPIO_Port GPIOH
#define IN2_4_Pin GPIO_PIN_0
#define IN2_4_GPIO_Port GPIOC
#define IN1_4_Pin GPIO_PIN_2
#define IN1_4_GPIO_Port GPIOC
#define ENCB_3_Pin GPIO_PIN_1
#define ENCB_3_GPIO_Port GPIOA
#define PWM_3_Pin GPIO_PIN_2
#define PWM_3_GPIO_Port GPIOA
#define PWM_4_Pin GPIO_PIN_3
#define PWM_4_GPIO_Port GPIOA
#define ENCA_3_Pin GPIO_PIN_5
#define ENCA_3_GPIO_Port GPIOA
#define ENCA_4_Pin GPIO_PIN_6
#define ENCA_4_GPIO_Port GPIOA
#define ENCB_4_Pin GPIO_PIN_7
#define ENCB_4_GPIO_Port GPIOA
#define ENCA_1_Pin GPIO_PIN_9
#define ENCA_1_GPIO_Port GPIOE
#define ENCB_1_Pin GPIO_PIN_11
#define ENCB_1_GPIO_Port GPIOE
#define LD4_Pin GPIO_PIN_12
#define LD4_GPIO_Port GPIOD
#define LD3_Pin GPIO_PIN_13
#define LD3_GPIO_Port GPIOD
#define LD5_Pin GPIO_PIN_14
#define LD5_GPIO_Port GPIOD
#define LD6_Pin GPIO_PIN_15
#define LD6_GPIO_Port GPIOD
#define VBUS_FS_Pin GPIO_PIN_9
#define VBUS_FS_GPIO_Port GPIOA
#define OTG_FS_ID_Pin GPIO_PIN_10
#define OTG_FS_ID_GPIO_Port GPIOA
#define OTG_FS_DM_Pin GPIO_PIN_11
#define OTG_FS_DM_GPIO_Port GPIOA
#define OTG_FS_DP_Pin GPIO_PIN_12
#define OTG_FS_DP_GPIO_Port GPIOA
#define SWDIO_Pin GPIO_PIN_13
#define SWDIO_GPIO_Port GPIOA
#define SWCLK_Pin GPIO_PIN_14
#define SWCLK_GPIO_Port GPIOA
#define IN1_1_Pin GPIO_PIN_15
#define IN1_1_GPIO_Port GPIOA
#define IN2_1_Pin GPIO_PIN_11
#define IN2_1_GPIO_Port GPIOC
#define IN1_2_Pin GPIO_PIN_2
#define IN1_2_GPIO_Port GPIOD
#define Audio_RST_Pin GPIO_PIN_4
#define Audio_RST_GPIO_Port GPIOD
#define OTG_FS_OverCurrent_Pin GPIO_PIN_5
#define OTG_FS_OverCurrent_GPIO_Port GPIOD
#define IN2_2_Pin GPIO_PIN_6
#define IN2_2_GPIO_Port GPIOD
#define SWO_Pin GPIO_PIN_3
#define SWO_GPIO_Port GPIOB
#define ENCA_2_Pin GPIO_PIN_6
#define ENCA_2_GPIO_Port GPIOB
#define ENCB_2_Pin GPIO_PIN_7
#define ENCB_2_GPIO_Port GPIOB
#define PWM_2_Pin GPIO_PIN_8
#define PWM_2_GPIO_Port GPIOB
#define IN1_3_Pin GPIO_PIN_9
#define IN1_3_GPIO_Port GPIOB
#define MEMS_INT2_Pin GPIO_PIN_1
#define MEMS_INT2_GPIO_Port GPIOE

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
