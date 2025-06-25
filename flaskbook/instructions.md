# Flaskbook 웹 애플리케이션 구현 가이드

## 프로젝트 개요
Flaskbook은 Flask 기반의 실시간 자세 분석 웹 애플리케이션입니다. MediaPipe를 활용하여 웹캠으로 촬영된 사용자의 자세를 실시간으로 분석하고, 로그인 시스템을 통해 개인별 분석 결과를 데이터베이스에 저장하여 목, 척추, 어깨, 골반 등의 자세 상태를 평가하고 기록을 관리합니다.

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
4. Blueprint 등록 ('/crud', '/auth' 경로)
5. 정적 파일 경로 설정
6. 세션 관리 설정
```

**핵심 기능:**
- 애플리케이션 팩토리 패턴으로 모듈화된 구조
- SQLite 데이터베이스 연결 설정
- Blueprint를 통한 라우팅 모듈화 (crud, auth)
- 세션 기반 인증 시스템

### 2. 데이터베이스 모델

#### `apps/crud/models.py` - 데이터 모델
```python
# 주요 구현 내용:
1. User 모델 클래스
   - 사용자 정보 (username, email, password_hash)
   - 로그인 시간 기록 (last_login)
   - 비밀번호 해싱 및 검증
   - PostureRecord와의 관계 설정

2. PostureRecord 모델 클래스
   - 자세 분석 결과 저장
   - 각 부위별 분석 데이터 (목, 척추, 어깨, 골반, 척추 틀어짐)
   - 종합 점수 계산 (0-100점)
   - 등급 계산 (A, B, C, D)
   - 분석 시간 기록
```

**핵심 기능:**
- **비밀번호 보안**: Werkzeug를 이용한 해싱
- **관계 설정**: User ↔ PostureRecord (1:N)
- **자동 점수 계산**: 분석 결과를 바탕으로 종합 점수 산출
- **등급 시스템**: 점수 기반 A-D 등급 자동 계산

### 3. 인증 시스템

#### `apps/crud/auth.py` - 인증 라우트
```python
# 주요 구현 내용:
1. 회원가입 ('/register', POST)
   - 사용자 정보 유효성 검사
   - 중복 사용자명/이메일 검사
   - 비밀번호 해싱 및 저장

2. 로그인 ('/login', POST)
   - 사용자명/비밀번호 검증
   - 세션 생성 및 로그인 시간 기록
   - 로그인 성공 시 메인 페이지로 리다이렉트

3. 로그아웃 ('/logout')
   - 세션 정리

4. 프로필 ('/profile')
   - 사용자 정보 및 분석 통계 표시
   - 최근 분석 기록 조회
```

**핵심 기능:**
- **세션 기반 인증**: Flask 세션을 이용한 사용자 인증
- **보안**: 비밀번호 해싱, 중복 검사
- **사용자 경험**: 플래시 메시지를 통한 피드백
- **데이터 분리**: 사용자별 데이터 격리

### 4. 자세 분석 모듈

#### `apps/crud/views.py` - API 엔드포인트
```python
# 주요 구현 내용:
1. 로그인 데코레이터 (@login_required)
2. 메인 페이지 라우트 ('/') - 인증 필요
3. 자세 분석 API ('/analyze', POST) - 인증 필요
4. 분석 기록 페이지 ('/history') - 인증 필요
5. 통계 페이지 ('/statistics') - 인증 필요
6. 데이터베이스 저장 로직
```

**핵심 기능:**
- **인증 보호**: 모든 페이지에 로그인 필요
- **실시간 분석**: 웹캠 프레임을 받아서 MediaPipe로 자세 분석
- **데이터 저장**: 모든 분석 결과를 데이터베이스에 저장
- **종합 점수**: 실시간으로 계산된 점수와 등급 반환

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

### 5. 프론트엔드 구현

#### `apps/templates/crud/index.html` - 메인 페이지
```html
# 주요 구현 내용:
1. 네비게이션 바 (분석, 기록, 통계, 프로필, 로그아웃)
2. 반응형 레이아웃 (flexbox)
3. 웹캠 비디오 요소
4. 종합 점수 표시 영역
5. 분석 결과 표시 영역
6. 카메라 ON/OFF 버튼
7. 숨겨진 캔버스 (이미지 처리용)
```

**UI 구성:**
- **상단**: 네비게이션 바 (사용자 인증 상태 표시)
- **좌측**: 웹캠 영상 + 카메라 제어 버튼
- **우측**: 종합 점수 + 실시간 분석 결과 표시

#### `apps/templates/crud/login.html` - 로그인 페이지
```html
# 주요 구현 내용:
1. 그라디언트 배경 디자인
2. 로그인 폼 (사용자명, 비밀번호)
3. 플래시 메시지 표시
4. 회원가입 링크
5. 반응형 디자인
```

**UI 특징:**
- **모던 디자인**: 그라디언트 배경과 카드 스타일
- **사용자 피드백**: 플래시 메시지로 오류/성공 알림
- **접근성**: 명확한 라벨과 포커스 상태

#### `apps/crud/static/crud/script.js` - 클라이언트 로직
```javascript
# 주요 구현 내용:
1. 웹캠 권한 요청 및 스트림 관리
2. 실시간 프레임 캡처 (1초 간격)
3. Canvas를 통한 이미지 처리
4. FormData를 이용한 서버 전송
5. 종합 점수 및 등급 표시
6. 분석 결과 실시간 표시
```

**핵심 기능:**
- **카메라 제어**: `toggleCamera()` 함수로 ON/OFF
- **프레임 캡처**: Canvas에 비디오 프레임 그리기
- **서버 통신**: Blob 형태로 이미지 전송
- **점수 표시**: 종합 점수와 등급을 실시간으로 표시
- **결과 파싱**: JSON 응답을 파싱하여 UI 업데이트

### 6. 실행 스크립트

#### `init_db.py` - 데이터베이스 초기화
```python
# 주요 구현 내용:
1. 데이터베이스 테이블 생성
2. 테스트 사용자 생성 (선택사항)
3. 초기화 완료 메시지
```

#### `run.py` - 애플리케이션 실행
```python
# 주요 구현 내용:
1. Flask 애플리케이션 실행
2. 접속 URL 안내
3. 디버그 모드 활성화
```

## 웹 애플리케이션 동작 흐름

### 1. 초기 접속 및 인증
1. 사용자가 웹페이지 접속 (`/auth/login`)
2. 로그인 페이지 로드
3. 사용자명/비밀번호 입력 및 제출
4. 서버에서 인증 검증
5. 성공 시 세션 생성 및 메인 페이지로 리다이렉트

### 2. 메인 페이지 접속
1. 로그인 데코레이터로 인증 상태 확인
2. 인증되지 않은 경우 로그인 페이지로 리다이렉트
3. 인증된 경우 메인 분석 페이지 로드
4. 사용자 정보 및 네비게이션 표시

### 3. 카메라 활성화
1. 사용자가 "카메라 ON/OFF" 버튼 클릭
2. `getUserMedia()` API로 카메라 스트림 요청
3. 비디오 요소에 스트림 연결
4. 분석 시작 (`startAnalyzing()`)

### 4. 실시간 분석 및 저장
1. 1초마다 비디오 프레임을 Canvas에 캡처
2. Canvas를 Blob으로 변환
3. FormData에 이미지 첨부하여 `/crud/analyze`로 POST 요청
4. 서버에서 MediaPipe로 자세 분석
5. **분석 결과를 데이터베이스에 저장**
6. **종합 점수 및 등급 계산**
7. JSON 형태로 분석 결과 반환
8. 클라이언트에서 결과를 파싱하여 UI 업데이트

### 5. 분석 결과 표시
```
종합 점수: 85점 (등급: B)

상세 분석 결과:
[목] 양호한 자세 (8.5°)
[척추] 정상
[어깨] 정상
[골반] 정상
[척추 틀어짐] 정상

분석 시간: 2024-01-15 14:30:25
```

## 데이터베이스 스키마

### User 테이블
```sql
CREATE TABLE user (
    id INTEGER PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME
);
```

### PostureRecord 테이블
```sql
CREATE TABLE posture_record (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    analysis_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    neck_angle FLOAT,
    neck_grade VARCHAR(1),
    neck_description VARCHAR(100),
    spine_is_hunched BOOLEAN,
    spine_angle FLOAT,
    shoulder_is_asymmetric BOOLEAN,
    shoulder_height_difference FLOAT,
    pelvic_is_tilted BOOLEAN,
    pelvic_angle FLOAT,
    spine_is_twisted BOOLEAN,
    spine_alignment FLOAT,
    overall_score INTEGER,
    overall_grade VARCHAR(1),
    FOREIGN KEY (user_id) REFERENCES user (id)
);
```

## 기술적 특징

### 1. 보안
- **비밀번호 해싱**: Werkzeug의 `generate_password_hash` 사용
- **세션 관리**: Flask 세션을 통한 사용자 인증
- **데이터 분리**: 사용자별 데이터 격리

### 2. 실시간 처리
- 1초 간격으로 프레임 분석
- 비동기 HTTP 요청으로 서버 통신
- 실시간 UI 업데이트 및 점수 표시

### 3. 컴퓨터 비전
- MediaPipe Pose 모델 활용
- 33개 신체 랜드마크 추출
- 기하학적 각도 계산으로 자세 평가

### 4. 웹 기술
- HTML5 Video API
- Canvas API
- Fetch API
- FormData
- CSS3 그라디언트 및 반응형 디자인

### 5. 서버 아키텍처
- Flask Blueprint 패턴
- RESTful API 설계
- JSON 기반 데이터 교환
- SQLAlchemy ORM

## 사용 방법

### 1. 환경 설정
```bash
cd flaskbook
pip install -r requirements.txt
```

### 2. 데이터베이스 초기화
```bash
python init_db.py
```

### 3. 애플리케이션 실행
```bash
python run.py
```

### 4. 웹 브라우저 접속
```
http://localhost:5000/auth/login
```

## 확장 가능성

### 1. 데이터 분석
- **트렌드 분석**: 장기간 자세 변화 추적
- **개인화**: 개인별 자세 패턴 분석
- **통계 시각화**: 차트 및 그래프 추가

### 2. 음성 피드백
- **gTTS 활용**: 실시간 음성 안내
- **자세 교정 가이드**: 음성으로 자세 개선 방법 안내

### 3. UI/UX 개선
- **모바일 최적화**: 반응형 디자인 강화
- **다크 모드**: 사용자 선호도에 따른 테마 변경
- **애니메이션**: 부드러운 전환 효과

### 4. 고도화된 분석
- **머신러닝 통합**: 개인화된 자세 평가 모델
- **실시간 피드백**: 즉시 자세 교정 안내
- **운동 추천**: 개인별 맞춤 운동 프로그램

### 5. 소셜 기능
- **친구 초대**: 가족/친구와 함께 자세 개선
- **랭킹 시스템**: 자세 개선 경쟁 요소
- **공유 기능**: 성과를 소셜 미디어에 공유 