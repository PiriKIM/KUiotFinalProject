#include "esp_camera.h"
#include <WiFi.h>
#include <esp_http_server.h>

// Wi-Fi 설정
const char* ssid = "turtle";
const char* password = "turtlebot3";

// AI Thinker ESP32-CAM 핀맵
#define PWDN_GPIO_NUM     32  // ✅ AI Thinker는 32번
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// MJPEG 스트리밍 핸들러 정의
void startCameraServer();

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  // Wi-Fi 연결
  WiFi.begin(ssid, password);
  Serial.println("Wi-Fi 연결 중...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi 연결 완료!");
  Serial.print("🔗 접속 주소: http://");
  Serial.println(WiFi.localIP());

  // 카메라 설정
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_CIF;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // 카메라 초기화
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ 카메라 초기화 실패! 오류 코드: 0x%x\n", err);
    Serial.println("👉 카메라 모듈이 제대로 꽂혀있는지 확인하거나, 전원 공급을 점검하세요.");
    return;
  }

  // MJPEG 스트리밍 서버 시작
  startCameraServer();
  Serial.println("📸 MJPEG 스트림 시작됨!");
  Serial.print("🔗 접속 주소: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
}

void loop() {
  delay(100);
}

// MJPEG 서버 핸들러
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;

  httpd_handle_t stream_httpd = NULL;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = [](httpd_req_t *req) {
      camera_fb_t * fb = NULL;
      esp_err_t res = ESP_OK;

      res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");

      while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
          Serial.println("❌ 프레임 획득 실패");
          continue;
        }

        char buf[64];
        snprintf(buf, sizeof(buf),
                 "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
        res = httpd_resp_send_chunk(req, buf, strlen(buf));
        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        res = httpd_resp_send_chunk(req, "\r\n", 2);

        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;
      }

      return res;
    },
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("📡 스트리밍 서버 시작됨.");
  } else {
    Serial.println("❌ 스트리밍 서버 시작 실패.");
  }
}
