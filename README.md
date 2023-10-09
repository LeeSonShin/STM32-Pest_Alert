## STM32-Pest_Alert : TensorFlow Lite model on STM32
> 본 프로젝트는 STM32CubeIDE에서 생성한 TinyML 프로젝트이다. 해충 판별 TinyML모델을 이용하여 STM32F746G-DISCO 보드에서 edge computing을 통해 해충을 판별하며, 블루투스 통신을 통해 연결된 안드로이드 애플리케이션으로 탐지된 해충의 종류를 전송한다.
### Hardware Information
- 개발 보드 : [STM32F746G-DISCO](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
- 카메라 모듈 : [Arducam SPI Camera(OV2640)](https://www.arducam.com/spi-arduino-camera/)
- 블루투스 모듈 : [FB755AC](https://www.devicemart.co.kr/goods/view?no=12419)
### Version Information
- STM32CubeIDE : 1.5
- Python : 3.7
- Tensorflow : 2.12.1
### Directory Structure
```
/
|- /Core
    |- /Startup
|- /Drivers
    |- /BSP
    |- /CMSIS
    |- /STM327xx_HAL_Driver
|- /tensorflow
    |- /signal
    |- /tensorflow
        |- /lite
            |- /c
            |- /core
            |- /kernels
            |- /micro
            |- /schema
|- /Utilities
    |- /Fonts
```
- `main.cpp`는 `Core`폴더 내부 있다.
- `main.cpp`에서는 아래와 같은 동작을 한다.
    - model interpreter 선언 및 호출
    - Arducam Camera에서 전송된 이미지 데이터 인덱싱
    - 디버깅을 위한 lcd 제어
    - model inference
    - 블루투스 통신을 활용하여 model output을 외부로 전송
### Related Works
- [Pest_Alert Android Application](https://github.com/LeeSonShin/Pest_Alert_Application)
- [Model Maker Notebook](https://github.com/LeeSonShin/Model_Maker)
### Reference
- [TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers](https://www.amazon.com/TinyML-Learning-TensorFlow-Ultra-Low-Power-Microcontrollers/dp/1492052043)
- [tinyengine](https://github.com/mit-han-lab/tinyengine) (Arducam Driver)
- [TensorFlow Lite For Microcontrollers](https://www.tensorflow.org/lite/microcontrollers?hl=ko)
