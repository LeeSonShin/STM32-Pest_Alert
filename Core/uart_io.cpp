#include "stm32f7xx_hal.h"

extern UART_HandleTypeDef DebugUartHandler;

extern "C" int _write(int file, char *ptr, int len) {
    for (int i = 0; i < len; i++) {
        HAL_UART_Transmit(&DebugUartHandler, (uint8_t *)&ptr[i], 1, HAL_MAX_DELAY);
    }
    return len;
}
