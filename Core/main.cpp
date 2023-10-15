/**
  ******************************************************************************
  * @file    main.cpp
  * @author  LeeSonShin
  * @brief   This file provides main program functions
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include <lcd_sine.h>
#include "stm32746g_discovery.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "saved_model.h"
#include "camera.h"
#include "main.h"
#include "lcd.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
# define RES_W 96
# define RES_H 96
//# define OUTPUT_CH 2
# define BUTTON1_Pin GPIO_PIN_0
# define BUTTON1_GPIO_Port GPIOA
# define BUTTON2_Pin GPIO_PIN_10
# define BUTTON2_GPIO_Port GPIOF

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/*  해상도에 따른 버퍼 크기 설정
* -------------------------
* | 80  * 80   : 44800    |
* | 96  * 96   : 64512	  |
* | 128 * 128 : 114688    | */
 static signed char buffer[64512];

//signed char out_int[OUTPUT_CH];
uint16_t *RGBbuf;

namespace {
	const tflite::Model * model = nullptr;
	tflite::MicroInterpreter * interpreter = nullptr;
	TfLiteTensor * model_input = nullptr;
	TfLiteTensor * model_output = nullptr;

	constexpr int kTensorArenaSize = 230 * 1024;
	alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// NOTE: extern is used because lcd.c also uses this variable.

// UART handler declaration
UART_HandleTypeDef DebugUartHandler;
UART_HandleTypeDef huart6;

/* Private function prototypes -----------------------------------------------*/
static void error_handler(void);
static void uart1_init(void);
void handle_output(uint8_t beeScore, uint8_t butterflyScore, uint8_t mothScore, uint8_t stinkScore, uint8_t noneScore);
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
    /* Enable the CPU Cache */
	CPU_CACHE_Enable();

    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* Configure the system clock */
    SystemClock_Config();

   /* Initialize all configured peripherals */
    MX_GPIO_Init();
    BSP_PB_Init(BUTTON_KEY, BUTTON_MODE_GPIO);

    /* Initialize UART1 */
    uart1_init();
    MicroPrintf("Point_1");
    /* Initialize LCD */
    lcdsetup();
    /* TfLite Initialization Code */
    model = tflite::GetModel(saved_model);
	if(model->version() != TFLITE_SCHEMA_VERSION) {
		MicroPrintf(
			"Model provided is schema verison %d not equal "
			"to supported version %d.",
			model->version(), TFLITE_SCHEMA_VERSION);
		return 0;
	}
	MicroPrintf("Point_2");
	static tflite::MicroMutableOpResolver<17> micro_op_resolver;
	micro_op_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
	micro_op_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());
	micro_op_resolver.AddDepthwiseConv2D(
	  tflite::Register_DEPTHWISE_CONV_2D_INT8());
	micro_op_resolver.AddReshape();
	micro_op_resolver.AddShape();
	micro_op_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());
	micro_op_resolver.AddMaxPool2D(tflite::Register_MAX_POOL_2D_INT8());
	micro_op_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());
	micro_op_resolver.AddStridedSlice();
	micro_op_resolver.AddPack();
	micro_op_resolver.AddPad(tflite::Register_PAD());
	micro_op_resolver.AddAdd(tflite::Register_ADD_INT8());
	micro_op_resolver.AddMean();
	micro_op_resolver.AddQuantize();
	micro_op_resolver.AddHardSwish();
	micro_op_resolver.AddMul(tflite::Register_MUL());  //INT8()이 있는데 인식 안됨;

	micro_op_resolver.AddConcatenation();



	static tflite::MicroInterpreter static_interpreter(
			model, micro_op_resolver, tensor_arena, kTensorArenaSize);
	interpreter = &static_interpreter;

	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if(allocate_status != kTfLiteOk) {
		MicroPrintf("AllocateTensors() failed");
		return 0;
	}
	MicroPrintf("Point_3");

  	// Obtain pointers to the model's input and output tensors.
  	model_input = interpreter->input(0);
  	model_output = interpreter->output(0);

  	// Index for model_output
	const int beeIndex = 0;
	const int butterflyIndex = 1;
	const int mothIndex = 2;
	const int stinkIndex = 4;
	const int noneIndex = 3;

    // We are dividing the whole input range with the number of inference
    // per cycle we want to show to get the unit value. We will then multiply
    // the unit value with the current position of the inference

    /* Arducam Camera Setup */
    int camErr = initCamera();
    if(camErr != 0) {
    	/* initCamera() Failed */
    	return 0;
    }

    StartCapture();
    signed char * input = getInput();
    RGBbuf = (uint16_t *)&input[RES_H * RES_W * 4];

    /* bluetooth setup */
    MX_USART6_UART_Init();

    /* Infinity while loop */
    while (1)
    {
        /* Camera Read */
        ReadCapture();
        StartCapture();

        DecodeandProcessAndRGB(RES_W, RES_H, input, RGBbuf, 1);

        // input tensor에 카메라 버퍼의 값 쓰기
        for (int i = 0; i < RES_W; i++) {
          for (int j = 0; j < RES_W; j++) {
        	int index = (i * RES_W + j) * 3;
            uint8_t red = (int32_t)input[index] + 128;
            uint8_t green = (int32_t)input[index + 1] + 128;
            uint8_t blue = (int32_t)input[index + 2] + 128;

            // for gray scale image
//            uint8_t gray = (int32_t)(red * 0.2999 + 0.587 * green + 0.114 * blue);


            model_input->data.uint8[index] = red;
            model_input->data.uint8[index + 1] = green;
            model_input->data.uint8[index + 2] = blue;

//            model_input->data.uint8[index] = gray;

            uint16_t b = (blue >> 3) & 0x1f;
            uint16_t g = ((green >> 2) & 0x3f) << 5;
            uint16_t r = ((red >> 3) & 0x1f) << 11;

            RGBbuf[j + RES_W * i] = (uint16_t)(r | g | b);
          }
        }
        loadRGB565LCD(10, 10, RES_W, RES_W, RGBbuf, 2);

		// Run inference, and report any error
		TfLiteStatus invoke_status = interpreter->Invoke();
		if (invoke_status != kTfLiteOk)
		{
			MicroPrintf("Invoke failed");
			return 0;
		}

		// 블루투스 통신을 이용하여 모델 출력을 애플리케이션으로 전송
		handle_output(model_output->data.uint8[beeIndex],
					  model_output->data.uint8[butterflyIndex],
					  model_output->data.uint8[mothIndex],
					  model_output->data.uint8[stinkIndex],
					  model_output->data.uint8[noneIndex]);
    }
}


void handle_output(uint8_t beeScore, uint8_t butterflyScore, uint8_t mothScore, uint8_t stinkScore, uint8_t noneScore)
{
	uint8_t bee_msg[6] = "0";
	uint8_t butterfly_msg[12] = "1";
	uint8_t moth_msg[7] = "2";
	uint8_t stink_msg[8] = "4";
	uint8_t none_msg[7] = "3";

	int max;
	max = (beeScore > butterflyScore) ? beeScore : butterflyScore;
	max = (max > mothScore) ? max : mothScore;
	max = (max > stinkScore) ? max : stinkScore;
	max = (max > noneScore) ? max : noneScore;
	char showbuf[150];
	//MicroPrintf("Bee: %d, Butterfly: %d, Moth: %d, Stink: %d, None: %d\r\n", beeScore, butterflyScore, mothScore, stinkScore, noneScore);
	if(max > noneScore) {
		if(max == beeScore) {
			HAL_UART_Transmit(&huart6, bee_msg, 1, 6);
			HAL_UART_Transmit(&DebugUartHandler, bee_msg, 1, 6);
			sprintf(showbuf, "BEE      ");
			displaystring(showbuf, 273, 10);

		}
		else if(max == butterflyScore) {
			HAL_UART_Transmit(&huart6, butterfly_msg, 1, 12);
			HAL_UART_Transmit(&DebugUartHandler, butterfly_msg, 1, 6);
			sprintf(showbuf, "BUTTERFLY ");
			displaystring(showbuf, 273, 10);
		}
		else if(max == mothScore) {
			HAL_UART_Transmit(&huart6, moth_msg, 1, 7);
			HAL_UART_Transmit(&DebugUartHandler, moth_msg, 1, 6);
			sprintf(showbuf, "MOTH      ");
			displaystring(showbuf, 273, 10);
		}
		else if(max == stinkScore) {
			HAL_UART_Transmit(&huart6, stink_msg, 1, 8);
			HAL_UART_Transmit(&DebugUartHandler, stink_msg, 1, 6);
			sprintf(showbuf, "STINK    ");
			displaystring(showbuf, 273, 10);
		}
	}
	else {
		HAL_UART_Transmit(&DebugUartHandler, none_msg, 1, 7);
		sprintf(showbuf, "           ");
		displaystring(showbuf, 273, 10);
	}
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
	DebugUartHandler.Init.BaudRate   = 115200;
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
