# Flaskbook 웹 애플리케이션 구현 가이드

## 프로젝트 개요
Flaskbook은 Flask 기반의 실시간 자세 분석 웹 애플리케이션입니다. MediaPipe를 활용하여 웹캠으로 촬영된 사용자의 자세를 실시간으로 분석하고, 목, 척추, 어깨, 골반 등의 자세 상태를 평가합니다.

## 구현 아키텍처

### 1. 애플리케이션 초기화 및 설정

#### `requirements.txt`
```
Flask>=2.0              # 웹 프레임워크
Flask-SQLAlchemy>=3.0   # ORM
python-dotenv>=0.21     # 환경변수 관리
gTTS>=2.3               # Google TTS (음성 피드백용)
opencv-python>=4.5.0    # 이미지 처리
```

#### `apps/app.py` - 애플리케이션 팩토리
```python
# 주요 구현 내용:
1. Flask 애플리케이션 팩토리 패턴 구현
2. SQLAlchemy 데이터베이스 설정 (SQLite)
3. Flask-Migrate 연동
4. Blueprint 등록 ('/crud' 경로)
5. 정적 파일 경로 설정
```

**핵심 기능:**
- 애플리케이션 팩토리 패턴으로 모듈화된 구조
- SQLite 데이터베이스 연결 설정
- Blueprint를 통한 라우팅 모듈화

### 2. 자세 분석 모듈

#### `apps/crud/views.py` - API 엔드포인트
```python
# 주요 구현 내용:
1. Blueprint 정의 ('crud')
2. 메인 페이지 라우트 ('/')
3. 자세 분석 API ('/analyze', POST)
4. MediaPipe Pose 모델 초기화
5. 이미지 처리 및 랜드마크 추출
```

**핵심 기능:**
- 웹캠 프레임을 받아서 MediaPipe로 자세 분석
- JSON 형태로 분석 결과 반환
- 실시간 이미지 처리 (OpenCV + NumPy)

#### `apps/crud/neck.py` - 자세 분석 알고리즘
```python
# 주요 구현 내용:
1. PostureAnalyzer 클래스
2. 각도 계산 함수 (calculate_angle)
3. 5가지 자세 분석 메서드:
   - analyze_turtle_neck_detailed(): 목 자세 분석
   - analyze_spine_curvature(): 척추 곡률 분석
   - analyze_shoulder_asymmetry(): 어깨 비대칭 분석
   - analyze_pelvic_tilt(): 골반 기울기 분석
   - analyze_spine_twisting(): 척추 틀어짐 분석
```

**분석 기준:**
- **목 자세**: 5° 이하=A, 10° 이하=B, 15° 이하=C, 15° 초과=D
- **척추 곡률**: 12° 초과 시 굽음으로 판정
- **어깨 비대칭**: 0.02 초과 시 비대칭으로 판정
- **골반 기울기**: 0.015 초과 시 기울어짐으로 판정
- **척추 틀어짐**: 0.03 초과 시 틀어짐으로 판정

#### `apps/crud/models.py` - 데이터 모델
```python
# 주요 구현 내용:
1. User 모델 클래스
2. 비밀번호 해싱 기능
3. 생성/수정 시간 자동 기록
```

**현재 상태:** 데이터베이스 모델은 정의되어 있지만 실제로는 사용되지 않음

### 3. 프론트엔드 구현

#### `apps/templates/crud/index.html` - 메인 페이지
```html
# 주요 구현 내용:
1. 반응형 레이아웃 (flexbox)
2. 웹캠 비디오 요소
3. 분석 결과 표시 영역
4. 카메라 ON/OFF 버튼
5. 숨겨진 캔버스 (이미지 처리용)
```

**UI 구성:**
- 좌측: 웹캠 영상 + 카메라 제어 버튼
- 우측: 실시간 분석 결과 표시

#### `apps/crud/static/crud/script.js` - 클라이언트 로직
```javascript
# 주요 구현 내용:
1. 웹캠 권한 요청 및 스트림 관리
2. 실시간 프레임 캡처 (1초 간격)
3. Canvas를 통한 이미지 처리
4. FormData를 이용한 서버 전송
5. 분석 결과 실시간 표시
```

**핵심 기능:**
- **카메라 제어**: `toggleCamera()` 함수로 ON/OFF
- **프레임 캡처**: Canvas에 비디오 프레임 그리기
- **서버 통신**: Blob 형태로 이미지 전송
- **결과 표시**: JSON 응답을 파싱하여 UI 업데이트

## 웹 애플리케이션 동작 흐름

### 1. 초기화 단계
1. 사용자가 웹페이지 접속 (`/crud/`)
2. HTML 페이지 로드 및 JavaScript 실행
3. 카메라 권한 사전 테스트

### 2. 카메라 활성화
1. 사용자가 "카메라 ON/OFF" 버튼 클릭
2. `getUserMedia()` API로 카메라 스트림 요청
3. 비디오 요소에 스트림 연결
4. 분석 시작 (`startAnalyzing()`)

### 3. 실시간 분석
1. 1초마다 비디오 프레임을 Canvas에 캡처
2. Canvas를 Blob으로 변환
3. FormData에 이미지 첨부하여 `/crud/analyze`로 POST 요청
4. 서버에서 MediaPipe로 자세 분석
5. JSON 형태로 분석 결과 반환
6. 클라이언트에서 결과를 파싱하여 UI 업데이트

### 4. 분석 결과 표시
```
[목] 양호한 자세 (8.5°)
[척추] 정상
[어깨] 정상
[골반] 정상
[척추 틀어짐] 정상
```

## 기술적 특징

### 1. 실시간 처리
- 1초 간격으로 프레임 분석
- 비동기 HTTP 요청으로 서버 통신
- 실시간 UI 업데이트

### 2. 컴퓨터 비전
- MediaPipe Pose 모델 활용
- 33개 신체 랜드마크 추출
- 기하학적 각도 계산으로 자세 평가

### 3. 웹 기술
- HTML5 Video API
- Canvas API
- Fetch API
- FormData

### 4. 서버 아키텍처
- Flask Blueprint 패턴
- RESTful API 설계
- JSON 기반 데이터 교환

## 확장 가능성

### 1. 데이터 저장
- 분석 결과를 데이터베이스에 저장
- 사용자별 자세 이력 관리
- 통계 및 트렌드 분석

### 2. 음성 피드백
- gTTS를 활용한 음성 안내
- 실시간 자세 교정 가이드

### 3. UI/UX 개선
- 차트 및 그래프 시각화
- 자세 교정 운동 가이드
- 모바일 반응형 디자인

### 4. 고도화된 분석
- 머신러닝 모델 통합
- 개인화된 자세 평가 기준
- 장기간 자세 변화 추적 