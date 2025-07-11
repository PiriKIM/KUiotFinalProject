# 실시간 자세 분석 시스템 설정 가이드

ESP32-CAM과 Flask 웹서버를 연동한 실시간 자세 분석 시스템입니다.

## 🚀 시스템 구성

- **ESP32-CAM**: 영상 캡처 및 전송
- **Flask 웹서버**: 실시간 영상 수신 및 자세 분석
- **ML 모델**: 4way 자세 분류 모델 (정면/좌측면/우측면)
- **MediaPipe**: 실시간 자세 인식
- **SQLite**: 분석 결과 저장

## 📋 사전 요구사항

### 1. Python 환경 설정
```bash
cd flaskbook
pip install -r requirements.txt
```

### 2. ML 모델 확인
다음 파일들이 존재하는지 확인하세요:
- `/home/piri/KUiotFinalProject/pose_classifier_4way_model.pkl`
- `/home/piri/KUiotFinalProject/merge_mp/data/right_side_angle_analysis.csv`
- `/home/piri/KUiotFinalProject/merge_mp/data/left_side_angle_analysis.csv`

### 3. 데이터베이스 마이그레이션
```bash
cd flaskbook
python migrate_realtime.py
```

## 🔧 설정 방법

### 1. Flask 서버 설정

#### 서버 IP 주소 확인
```bash
# Linux/Mac
ifconfig

# Windows
ipconfig
```

#### ESP32-CAM 코드에서 서버 IP 수정
`piri/src/main.cpp` 파일에서 다음 부분을 수정하세요:

```cpp
// Flask 서버 설정
const char* serverUrl = "http://YOUR_SERVER_IP:5000/realtime/esp32_stream";
const char* analysisUrl = "http://YOUR_SERVER_IP:5000/realtime/get_analysis_result";
```

### 2. WiFi 설정

ESP32-CAM이 연결할 WiFi 정보를 `piri/src/main.cpp`에서 수정하세요:

```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
```

## 🚀 실행 방법

### 1. Flask 서버 실행
```bash
cd flaskbook
python run.py
```

서버가 `http://localhost:5000`에서 실행됩니다.

### 2. ESP32-CAM 업로드
```bash
cd piri
pio run --target upload
```

### 3. 웹 브라우저에서 접속
1. `http://localhost:5000/auth/login` - 로그인
2. `http://localhost:5000/realtime/realtime_analysis` - 실시간 분석 페이지

## 📱 사용 방법

### 실시간 분석 페이지
1. **분석 시작**: "분석 시작" 버튼 클릭
2. **실시간 스트림**: ESP32-CAM에서 전송된 영상 확인
3. **분석 결과**: 자세 등급(A/B/C) 및 피드백 메시지 확인
4. **분석 중지**: "분석 중지" 버튼 클릭
5. **기록 확인**: "분석 기록" 버튼으로 이전 분석 결과 확인

### 분석 결과
- **A등급**: 바른 자세 (초록색 LED)
- **B등급**: 보통 자세 (노란색 LED)
- **C등급**: 나쁜 자세 (빨간색 LED)

## 🔍 문제 해결

### 1. ESP32-CAM 연결 문제
- WiFi SSID/비밀번호 확인
- 서버 IP 주소 확인
- 시리얼 모니터로 연결 상태 확인

### 2. Flask 서버 오류
- 필요한 라이브러리 설치 확인
- ML 모델 파일 경로 확인
- 데이터베이스 마이그레이션 확인

### 3. 분석 결과가 나오지 않는 경우
- 카메라 앞에 서서 자세 취하기
- 측면을 향해 서기 (좌측면 또는 우측면)
- 충분한 조명 확인

## 📊 API 엔드포인트

### ESP32-CAM → Flask 서버
- `POST /realtime/esp32_stream`: 영상 데이터 전송

### 웹 클라이언트 → Flask 서버
- `GET /realtime/video_feed`: 실시간 비디오 스트림
- `POST /realtime/start_analysis`: 분석 시작
- `POST /realtime/stop_analysis`: 분석 중지
- `GET /realtime/get_analysis_result`: 분석 결과 조회
- `GET /realtime/analysis_history`: 분석 기록 조회

## 🗄️ 데이터베이스 스키마

### RealtimePostureRecord 테이블
- `id`: 기본 키
- `user_id`: 사용자 ID (외래 키)
- `timestamp`: 분석 시간
- `detected_side`: 감지된 측면 (front/left/right/unknown)
- `ml_confidence`: ML 모델 신뢰도
- `cva_angle`: CVA 각도
- `posture_grade`: 자세 등급 (A/B/C)
- `feedback_message`: 피드백 메시지
- `frame_count`: 프레임 번호

## 🔧 고급 설정

### 프레임 전송 간격 조정
ESP32-CAM에서 `FRAME_INTERVAL` 값을 수정하여 전송 간격을 조정할 수 있습니다:

```cpp
#define FRAME_INTERVAL 1000  // 1초 (밀리초 단위)
```

### JPEG 품질 조정
영상 품질과 전송 속도의 균형을 위해 `JPEG_QUALITY`를 조정할 수 있습니다:

```cpp
#define JPEG_QUALITY 10  // 1-63 (낮을수록 파일 크기 작음)
```

### LED 밝기 조정
```cpp
int brightness = 15;  // 0-255 (PWM 값)
```

## 📝 로그 확인

### ESP32-CAM 로그
시리얼 모니터에서 다음 정보를 확인할 수 있습니다:
- WiFi 연결 상태
- 프레임 캡처 상태
- 서버 응답
- 분석 결과

### Flask 서버 로그
터미널에서 다음 정보를 확인할 수 있습니다:
- ML 모델 로드 상태
- 분석기 초기화 상태
- 데이터베이스 저장 상태
- 오류 메시지

## 🎯 성능 최적화

1. **네트워크 최적화**: ESP32-CAM과 서버가 같은 WiFi 네트워크에 연결
2. **영상 품질**: 적절한 JPEG 품질 설정으로 전송 속도 최적화
3. **분석 주기**: 필요에 따라 분석 주기 조정
4. **메모리 관리**: 주기적인 데이터베이스 정리

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 모든 라이브러리가 올바르게 설치되었는지
2. ML 모델 파일이 올바른 경로에 있는지
3. 데이터베이스 마이그레이션이 완료되었는지
4. 네트워크 연결이 안정적인지 