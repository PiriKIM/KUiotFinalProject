#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// #define BUZZER_PIN 4  // GPIO4

#define LED_PIN 4  // 내장 LED (플래시)

// PWM 설정 값
#define PWM_CHANNEL 7
#define PWM_FREQ 5000
#define PWM_RESOLUTION 8  // 8비트 → 밝기 0~255

int brightness = 15;

// WiFi 설정
const char* ssid = "turtle";
const char* password = "turtlebot3";

// Flask 서버 설정
const char* serverUrl = "http://192.168.0.69:5000/realtime/esp32_stream";  // 실제 서버 IP로 변경 필요
const char* analysisUrl = "http://192.168.0.69:5000/realtime/get_analysis_result";  // 분석 결과 확인용

// 카메라 설정
#define FRAME_INTERVAL 1000  // 1초마다 프레임 전송
#define JPEG_QUALITY 10      // JPEG 품질 (낮을수록 파일 크기 작음)

void checkAnalysisResult();
void updateLEDStatus(String response);

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;  // 320x240
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = 1;

  esp_camera_init(&config);
}

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32-CAM 실시간 자세 분석 시스템 시작");

  // PWM 초기화
  ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(LED_PIN, PWM_CHANNEL);

  // WiFi 연결
  WiFi.begin(ssid, password);
  Serial.print("WiFi 연결 중");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi 연결됨");
  Serial.print("IP 주소: ");
  Serial.println(WiFi.localIP());

  // 카메라 초기화
  setupCamera();
  Serial.println("카메라 초기화 완료");

  // LED 테스트
  ledcWrite(PWM_CHANNEL, brightness);
  delay(1000);
  ledcWrite(PWM_CHANNEL, 0);
}

void loop() {
  // WiFi 연결 확인
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi 연결 끊어짐. 재연결 시도...");
    WiFi.reconnect();
    delay(5000);
    return;
  }

  // 카메라에서 프레임 캡처
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  Serial.printf("Frame captured: %d bytes\n", fb->len);

  // Flask 서버로 프레임 전송
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("Content-Length", String(fb->len));
  
  int httpResponseCode = http.POST(fb->buf, fb->len);

  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("서버 응답: %d - %s\n", httpResponseCode, response.c_str());
    
    // 분석 결과 확인 (선택사항)
    checkAnalysisResult();
    
    // LED 상태 업데이트 (분석 결과에 따라)
    updateLEDStatus(response);
  } else {
    Serial.printf("HTTP 요청 실패: %d\n", httpResponseCode);
    // 연결 실패 시 LED 깜빡임
    ledcWrite(PWM_CHANNEL, brightness);
    delay(100);
    ledcWrite(PWM_CHANNEL, 0);
  }

  http.end();
  esp_camera_fb_return(fb);

  // 프레임 간격 대기
  delay(FRAME_INTERVAL);
}

void checkAnalysisResult() {
  // 분석 결과를 별도로 확인 (선택사항)
  HTTPClient http;
  http.begin(analysisUrl);
  
  int httpResponseCode = http.GET();
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("분석 결과: %s\n", response.c_str());
    
    // JSON 파싱 시도
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, response);
    
    if (!error) {
      if (doc.containsKey("result")) {
        JsonObject result = doc["result"];
        if (result.containsKey("posture_grade")) {
          String grade = result["posture_grade"].as<String>();
          Serial.printf("자세 등급: %s\n", grade.c_str());
        }
      }
    }
  }
  
  http.end();
}

void updateLEDStatus(String response) {
  // 서버 응답에 따라 LED 상태 업데이트
  if (response.indexOf("A") >= 0) {
    // A등급: 초록색 LED (낮은 밝기)
    ledcWrite(PWM_CHANNEL, brightness / 3);
  } else if (response.indexOf("B") >= 0) {
    // B등급: 노란색 LED (중간 밝기)
    ledcWrite(PWM_CHANNEL, brightness * 2 / 3);
  } else if (response.indexOf("C") >= 0) {
    // C등급: 빨간색 LED (높은 밝기)
    ledcWrite(PWM_CHANNEL, brightness);
  } else {
    // 기타: LED 끄기
    ledcWrite(PWM_CHANNEL, 0);
  }
}
