/**
  ******************************************************************************
  * @file    main.cpp
  * @author  LeeSonShin
  * Code referenced from Fahad Mirza (fahadmirza8@gmail.com)
  * @brief   This file provides main program functions
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include <lcd_sine.h>
#include "stm32746g_discovery.h"
#include "sine_model.h"
#include "saved_model.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "camera.h"
#include "main.h"
#include "lcd.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
# define RES_W 80
# define RES_H 80
# define OUTPUT_CH 2
# define BUTTON1_Pin GPIO_PIN_0
# define BUTTON1_GPIO_Port GPIOA
# define BUTTON2_Pin GPIO_PIN_10
# define BUTTON2_GPIO_Port GPIOF

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
// for debuging
// 80 * 80 : 44800
// 128 * 128 : 114688
 static signed char buffer[44800]; // 177704 byte
// for test only
//static signed char buffer[1];

signed char out_int[OUTPUT_CH];
uint16_t *RGBbuf;

namespace
{
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* model_input = nullptr;
    TfLiteTensor* model_output = nullptr;

    // Create an area of memory to use for input, output, and intermediate arrays.
    // Finding the minimum value for your model may require some trial and error.
    // STM32F746G-DISCOVERY에서 할당 가능한 최댓값 : 32 * 21 * 1024 = 688_128
    // [2023-09-19] saved model 터진다... -> 텐서 아레나 크기를 최대로 당겨도 인터프리터에 텐서 아레나가 제대로 할당되지 않음
    constexpr uint32_t kTensorArenaSize = 2 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// This constant represents the range of x values our model was trained on,
// which is from 0 to (2 * Pi). We approximate Pi to avoid requiring
// additional libraries.
extern const float INPUT_RANGE = 2.f * 3.14159265359f;
// NOTE: extern is used because lcd.c also uses this variable.

// This constant determines the number of inferences to perform across the range
// of x values defined above. Since each inference takes time, the higher this
// number, the more time it will take to run through the entire range. The value
// of this constant can be tuned so that one full cycle takes a desired amount
// of time. Since different devices take different amounts of time to perform
// inference, this value should be defined per-device.
// A larger number than the default to make the animation smoother
const uint16_t INFERENCE_PER_CYCLE = 70;

// UART handler declaration
UART_HandleTypeDef DebugUartHandler;
UART_HandleTypeDef huart6;

/* Private function prototypes -----------------------------------------------*/
static void system_clock_config(void);
static void cpu_cache_enable(void);
static void error_handler(void);
static void uart1_init(void);
void handle_output(tflite::ErrorReporter* error_reporter, float x_value, float y_value);
signed char* getInput();
static void MX_GPIO_Init(void);
static void CPU_CACHE_Enable(void);
static void SystemClock_Config(void);
static void Error_Handler(void);

/* Private user code ---------------------------------------------------------*/
/**
  * @brief  The application entry point.
  * @retval int
  */
/** add for Bluetooth code **/


static void MX_USART6_UART_Init(void)
{

  /* USER CODE BEGIN USART6_Init 0 */

  /* USER CODE END USART6_Init 0 */

  /* USER CODE BEGIN USART6_Init 1 */

  /* USER CODE END USART6_Init 1 */
  huart6.Instance = USART6;
  huart6.Init.BaudRate = 9600;
  huart6.Init.WordLength = UART_WORDLENGTH_8B;
  huart6.Init.StopBits = UART_STOPBITS_1;
  huart6.Init.Parity = UART_PARITY_NONE;
  huart6.Init.Mode = UART_MODE_TX_RX;
  huart6.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart6.Init.OverSampling = UART_OVERSAMPLING_16;
  huart6.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart6.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart6) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART6_Init 2 */

  /* USER CODE END USART6_Init 2 */

}


int main(void)
{
	char buf[150];
	char showbuf[150];
    /* Enable the CPU Cache */
	CPU_CACHE_Enable();

    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* Configure the system clock */
    SystemClock_Config();

   /* Initialize all configured peripherals */
   /* 나중에 램 사용량 줄이기 위해 진짜 필요한 것 말고는 주석 처리 */
    MX_GPIO_Init();
    BSP_PB_Init(BUTTON_KEY, BUTTON_MODE_GPIO);

    /* Configure on-board green LED */
//    BSP_LED_Init(LED_GREEN);

    /* Initialize UART1 */
    uart1_init();

    /* Initialize LCD */
    lcdsetup();
    /* TfLite Initialization Code */
  	static tflite::MicroErrorReporter micro_error_reporter;
  	error_reporter = &micro_error_reporter;

  	// Map the model into a usable data structure. This doesn't involve any
  	// copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(sine_model);

  	if(model->version() != TFLITE_SCHEMA_VERSION)
  	{
  		TF_LITE_REPORT_ERROR(error_reporter,
  	                         "Model provided is schema version %d not equal "
  	                         "to supported version %d.",
  	                         model->version(), TFLITE_SCHEMA_VERSION);
  	    return 0;
  	}

  	// This pulls in all the operation implementations we need.
  	static tflite::ops::micro::AllOpsResolver resolver;

  	// Build an interpreter to run the model with.
  	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  	interpreter = &static_interpreter;
  	// Allocate memory from the tensor_arena for the model's tensors.
  	TfLiteStatus allocate_status = interpreter->AllocateTensors();
    // [2023-09-19] 텐서 아레나가 정상적으로 선언되지 않으면 아래 if문에서 return 되어버림.
  	if (allocate_status != kTfLiteOk)
  	{
  	    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
  	  	drawRedBackground(270, 480, 40, 100);
  	    return 0;
  	}
  	drawGreenBackground(270, 480, 40, 100);
  	// Obtain pointers to the model's input and output tensors.
  	model_input = interpreter->input(0);
  	model_output = interpreter->output(0);

    // We are dividing the whole input range with the number of inference
    // per cycle we want to show to get the unit value. We will then multiply
    // the unit value with the current position of the inference

  	// sine model code //
    // float unitValuePerDevision = INPUT_RANGE / static_cast<float>(INFERENCE_PER_CYCLE);


    /* Arducam Camera Setup */
    int camErr = initCamera();

    uint32_t start, end;
    StartCapture();
    signed char * input = getInput(); // signed char buffer[1084], input = buffer
    RGBbuf = (uint16_t *)&input[RES_H * RES_W * 4]; // input[25600]의 주소를 RGBbuf에 uint16_t의 포인터 형태로 저장
    int t_mode = 0;

    /* bluetooth setup */
    MX_USART6_UART_Init();
    uint8_t message[10] = "success\n";
    /* Infinity while loop */
    while (1)
    {
    	HAL_UART_Transmit(&huart6, message, 8, 10);
    	// HAL_UART_Transmit(&DebugUartHandler, &message, 1, 10);
//	    // Calculate an x value to feed into the model
//        for(uint16_t inferenceCount = 0; inferenceCount <= INFERENCE_PER_CYCLE; inferenceCount++)
//        {
//	        float x_val = static_cast<float>(inferenceCount) * unitValuePerDevision;
//
//	        // Place our calculated x value in the model's input tensor
//	        model_input->data.f[0] = x_val;
//
//	        // Run inference, and report any error
//	        TfLiteStatus invoke_status = interpreter->Invoke();
//	        if (invoke_status != kTfLiteOk)
//	        {
//	            TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x_val: %f\n", static_cast<float>(x_val));
//	            return 0;
//	        }
//
//	        // Read the predicted y value from the model's output tensor
//	        float y_val = model_output->data.f[0];
//
//	        // Do something with the results
//
//	        // handle_output(error_reporter, x_val, y_val);
//        }

        /* Camera Read */
        start = HAL_GetTick();
        ReadCapture();
        StartCapture();

        DecodeandProcessAndRGB(RES_W, RES_H, input, RGBbuf, 1);

        for (int i = 0; i < RES_W; i++) {
          for (int j = 0; j < RES_W; j++) {
            uint8_t red = (int32_t)input[(RES_W * i + j) * 3] + 128;
            uint8_t green = (int32_t)input[(RES_W * i + j) * 3 + 1] + 128;
            uint8_t blue = (int32_t)input[(RES_W * i + j) * 3 + 2] + 128;

            uint16_t b = (blue >> 3) & 0x1f;
            uint16_t g = ((green >> 2) & 0x3f) << 5;
            uint16_t r = ((red >> 3) & 0x1f) << 11;

            RGBbuf[j + RES_W * i] = (uint16_t)(r | g | b);
          }
        }
        loadRGB565LCD(10, 10, RES_W, RES_W, RGBbuf, 3);
    }
}


void handle_output(tflite::ErrorReporter* error_reporter, float x_value, float y_value)
{
	// Log the current X and Y values
	TF_LITE_REPORT_ERROR(error_reporter, "x_value: %f, y_value: %f\n", x_value, y_value);

	// A custom function can be implemented and used here to do something with the x and y values.
	// In my case I will be plotting sine wave on an LCD.
	LCD_Output(x_value, y_value);
}

signed char* getInput() { return &buffer[0]; }

/**
  * @brief  System Clock Configuration
  *         The system Clock is configured as follow :
  *            System Clock source            = PLL (HSE)
  *            SYSCLK(Hz)                     = 200000000
  *            HCLK(Hz)                       = 200000000
  *            AHB Prescaler                  = 1
  *            APB1 Prescaler                 = 4
  *            APB2 Prescaler                 = 2
  *            HSE Frequency(Hz)              = 25000000
  *            PLL_M                          = 25
  *            PLL_N                          = 400
  *            PLL_P                          = 2
  *            PLL_Q                          = 9
  *            VDD(V)                         = 3.3
  *            Main regulator output voltage  = Scale1 mode
  *            Flash Latency(WS)              = 6
  * @param  None
  * @retval None
  */
void system_clock_config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};


    // Configure the main internal regulator output voltage
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    // Enable HSE Oscillator and activate PLL with HSE as source
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState       = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM       = 25;
    RCC_OscInitStruct.PLL.PLLN       = 400;
    RCC_OscInitStruct.PLL.PLLP       = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ       = 9;

    if(HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
    	error_handler();
    }

    // Activate the Over-Drive mode
    if(HAL_PWREx_EnableOverDrive() != HAL_OK)
    {
    	error_handler();
    }

    // Initializes the CPU, AHB and APB busses clocks
    RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

    if(HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_6) != HAL_OK)
    {
        error_handler();
    }
}


/**
  * @brief  UART1 Initialization Function
  * @param  None
  * @retval None
  */
static void uart1_init(void)
{
    /*##-1- Configure the UART peripheral ######################################*/
	/* Put the USART peripheral in the Asynchronous mode (UART Mode)
	   UART configured as follows:
	      - Word Length = 8 Bits
	      - Stop Bit = One Stop bit
	      - Parity = None
	      - BaudRate = 9600 baud
	      - Hardware flow control disabled (RTS and CTS signals)
	 */

	DebugUartHandler.Instance        = DISCOVERY_COM1;
	DebugUartHandler.Init.BaudRate   = 9600;
	DebugUartHandler.Init.WordLength = UART_WORDLENGTH_8B;
	DebugUartHandler.Init.StopBits   = UART_STOPBITS_1;
	DebugUartHandler.Init.Parity     = UART_PARITY_NONE;
	DebugUartHandler.Init.HwFlowCtl  = UART_HWCONTROL_NONE;
	DebugUartHandler.Init.Mode       = UART_MODE_TX_RX;
	DebugUartHandler.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;

	if(HAL_UART_Init(&DebugUartHandler) != HAL_OK)
	{
	    error_handler();
	}
}


/**
  * @brief  This function is executed in case of error occurrence.
  * @param  None
  * @retval None
  */
static void error_handler(void)
{
    // Turn Green LED ON
    BSP_LED_On(LED_GREEN);
    while(1);
}

/**
  * @brief  CPU L1-Cache enable.
  * @param  None
  * @retval None
  */
static void cpu_cache_enable(void)
{
    // Enable I-Cache
    SCB_EnableICache();

    // Enable D-Cache
    SCB_EnableDCache();
}

static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOJ_CLK_ENABLE();
  __HAL_RCC_GPIOI_CLK_ENABLE();
  __HAL_RCC_GPIOK_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();

  HAL_GPIO_WritePin(OTG_FS_PowerSwitchOn_GPIO_Port, OTG_FS_PowerSwitchOn_Pin,
                    GPIO_PIN_SET);

  HAL_GPIO_WritePin(GPIOI, ARDUINO_D7_Pin | ARDUINO_D8_Pin, GPIO_PIN_RESET);

  HAL_GPIO_WritePin(LCD_BL_CTRL_GPIO_Port, LCD_BL_CTRL_Pin, GPIO_PIN_SET);

  HAL_GPIO_WritePin(LCD_DISP_GPIO_Port, LCD_DISP_Pin, GPIO_PIN_SET);

  HAL_GPIO_WritePin(DCMI_PWR_EN_GPIO_Port, DCMI_PWR_EN_Pin, GPIO_PIN_RESET);

  HAL_GPIO_WritePin(GPIOG, ARDUINO_D4_Pin | ARDUINO_D2_Pin | EXT_RST_Pin,
                    GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = OTG_HS_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_HS_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = RMII_TXD1_Pin | RMII_TXD0_Pin | RMII_TX_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ULPI_D7_Pin | ULPI_D6_Pin | ULPI_D5_Pin | ULPI_D3_Pin |
                        ULPI_D2_Pin | ULPI_D1_Pin | ULPI_D4_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_HS;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = SPDIF_RX0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF8_SPDIFRX;
  HAL_GPIO_Init(SPDIF_RX0_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = OTG_FS_VBUS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_FS_VBUS_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = Audio_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(Audio_INT_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = OTG_FS_P_Pin | OTG_FS_N_Pin | OTG_FS_ID_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_FS;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = OTG_FS_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(OTG_FS_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ARDUINO_D7_Pin | ARDUINO_D8_Pin | LCD_DISP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOI, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = uSD_Detect_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(uSD_Detect_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LCD_BL_CTRL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LCD_BL_CTRL_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = OTG_FS_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_FS_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = TP3_Pin | NC2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ARDUINO_SCK_D13_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
  HAL_GPIO_Init(ARDUINO_SCK_D13_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = DCMI_PWR_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(DCMI_PWR_EN_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_11;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOI, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LCD_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(LCD_INT_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ULPI_NXT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_HS;
  HAL_GPIO_Init(ULPI_NXT_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ARDUINO_D4_Pin | ARDUINO_D2_Pin | EXT_RST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ULPI_STP_Pin | ULPI_DIR_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_HS;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = RMII_MDC_Pin | RMII_RXD0_Pin | RMII_RXD1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = RMII_RXER_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(RMII_RXER_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = RMII_REF_CLK_Pin | RMII_MDIO_Pin | RMII_CRS_DV_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ULPI_CLK_Pin | ULPI_D0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_HS;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = ARDUINO_MISO_D12_Pin | ARDUINO_MOSI_PWM_D11_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = BUTTON1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(BUTTON1_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = BUTTON2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(BUTTON2_GPIO_Port, &GPIO_InitStruct);

  /* add for bluetooth */
  GPIO_InitStruct.Pin = GPIO_PIN_6;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}

void SystemClock_Config(void) {
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;
  HAL_StatusTypeDef ret = HAL_OK;

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;
  RCC_OscInitStruct.PLL.PLLN = 432;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 9;

  ret = HAL_RCC_OscConfig(&RCC_OscInitStruct);
  if (ret != HAL_OK) {
    while (1) {
      ;
    }
  }

  ret = HAL_PWREx_EnableOverDrive();
  if (ret != HAL_OK) {
    while (1) {
      ;
    }
  }

  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK |
                                 RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  ret = HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7);
  if (ret != HAL_OK) {
    while (1) {
      ;
    }
  }
}
static void Error_Handler(void) {

  BSP_LED_On(LED1);
  while (1) {
  }
}

static void CPU_CACHE_Enable(void) {

  SCB_EnableICache();

  SCB_EnableDCache();
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
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
}
#endif // USE_FULL_ASSERT
