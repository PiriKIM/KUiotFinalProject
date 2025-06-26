# CHANGELOG

## 2025-06-26

### 15:00
- User 모델에 password_hash 필드 및 비밀번호 해싱/검증 기능 추가 (`apps/crud/models.py`)
  ```python
  # 2024-07-05 15:00 변경
  password_hash = db.Column(db.String(255), nullable=False)
  @property
  def password(self):
      raise AttributeError("읽어 들일 수 없음")
  @password.setter
  def password(self, password):
      self.password_hash = generate_password_hash(password)
  def check_password(self, password):
      return check_password_hash(self.password_hash, password)
  ```

- PostureRecord 모델 및 종합 점수/등급 계산 기능 구현 (`apps/crud/models.py`)
  ```python
  # 2025-06-26 15:00 변경
  class PostureRecord(db.Model):
      ...
      def calculate_overall_score(self):
          ...
      def calculate_overall_grade(self):
          ...
  ```

### 15:10
- 회원가입/로그인/로그아웃/프로필 라우트 및 템플릿 추가 (`apps/crud/auth.py`, `templates/crud/`)
  - `/register`, `/login`, `/logout`, `/profile` 라우트 구현
  - 회원가입 시 중복 검사, 비밀번호 해싱, 플래시 메시지 등 추가
  - 프로필 페이지에서 최근 분석 기록, 통계 표시

### 15:20
- 자세 분석 결과 DB 저장 및 기록/통계/히스토리 기능 구현 (`apps/crud/views.py`)
  - `/analyze` 라우트에서 MediaPipe로 분석 후 PostureRecord에 저장
  - `/history` 라우트에서 사용자별 분석 기록 페이지네이션 제공
  - `/statistics` 라우트에서 통계(평균점수, 등급별 개수, 월별 통계 등) 제공

### 15:30
- 프론트엔드 템플릿 및 네비게이션 추가 (`templates/crud/`)
  - `index.html`, `login.html`, `register.html`, `profile.html`, `history.html`, `statistics.html` 등 구현
  - 실시간 분석, 기록/통계/프로필 UI 제공

### 15:40
- run.py 삭제됨 (서버 실행 스크립트)

---

### 2025-6-26

### 09:00
- html 파일 기존에 묶여있던 style css 코드 따로 css 파일 만들어서 분리
  -`apps/crud/static/analysis.css`
  -`apps/crud/static/auth.css`
  -`apps/crud/static/history.css`
  -`apps/crud/static/profile.css`
  -`apps/crud/static/statistics.css`
  -`apps/crud/static/style.css`

### 11:00
- 웹 페이지 카메라 화면에 관절 좌표 출력되도록 수정

  1. Flask 서버에서 landmarks 반환
    -수정 파일: `apps/crud/views.py`
    -수정 위치: `/analyze `라우트
  분석 결과를 반환할 때, landmarks 좌표 리스트도 함께 반환하도록 수정
  2. 프론트엔드 JS에서 시각화
    -수정 파일: `apps/crud/static/crud/script.js `(혹은 JS가 위치한 실제 파일명)
    -수정 위치: 서버에서 분석 결과(landmarks 등)를 받아 처리하는 부분 <canvas>에 landmarks, 분석선, 각도 등을 그리는 함수 추가
  3. HTML에서 video+canvas 구조
  수정 파일:
    -`apps/crud/templates/crud/index.html` (실시간 분석 메인 페이지)
    -수정 위치: <video> 태그가 있는 부분 <canvas> 태그를 추가해서 <video>와 겹치게 배치
  4. (선택) CSS
  수정 파일: `apps/crud/static/crud/style.css` (필요하다면 `canvas/video` 배치 스타일 조정)

## 2024-06-26

### flaskbook 주요 AI 변경 로그

- 오늘 flaskbook 폴더 내 주요 파일들의 변경 내역과 목적을 정리한 로그입니다.
- 작성자: AI (GPT-4)
- 날짜: 2024-06-26

#### 1. `apps/crud/views.py`
- **목적:** MediaPipe neck.py 방식에 맞춰 얼굴 랜드마크 제외, 상체 랜드마크만 추출하도록 분석 API 수정
- **주요 변경점:**
  - 얼굴(코, 눈 등) 제외, 귀(7번)부터 시작하는 상체 랜드마크만 추출
  - 프론트엔드로 상체 랜드마크만 반환
  - [AI 수정] 주석 추가로 변경 내역 명확히 표시

#### 2. `apps/templates/crud/index.html`
- **목적:**
  - 오른쪽 실시간 카메라 프레임 제거, 왼쪽 영상만 크게 표시
  - FPS(실제 렌더링 프레임) 표시, 분석 결과 UI 개선
- **주요 변경점:**
  - 오른쪽 camera-preview(비디오/캔버스) 완전 삭제
  - 왼쪽 영상 width/height 속성 제거, CSS로 비율 자동 조정
  - result-header에 FPS 표시용 `<div id="fps-indicator">` 추가

#### 3. `apps/crud/static/analysis.css`
- **목적:**
  - 왼쪽 영상(비디오/캔버스)만 크게, 비율(4:3) 유지
  - 오른쪽 프리뷰 관련 스타일 완전 삭제
- **주요 변경점:**
  - .video-container에 aspect-ratio: 4/3, object-fit: cover 적용
  - max-width: 800px, height 자동, background: #000
  - video/canvas의 width/height 속성 제거

#### 4. `apps/crud/static/crud/script.js`
- **목적:**
  - 실시간 웹캠 프레임(FPS) 측정 및 표시
  - 관절 좌표(landmarks)와 스켈레톤(연결선) 실시간 시각화
  - 목/어깨/골반/척추 중심점 및 척추선(목-어깨-골반) 시각화
  - 오른쪽 프리뷰 관련 코드 완전 삭제
- **주요 변경점:**
  - FPS 측정 및 표시 (실제 렌더링 기준)
  - requestAnimationFrame 기반 실시간 렌더링(renderLoop)
  - drawLandmarksAndSkeleton 함수로 landmarks, 스켈레톤, 중심점, 척추선 모두 시각화
  - 척추 중심선(목-어깨-골반) 파란색 두꺼운 선 추가
  - 기존 drawLandmarks 함수 삭제, drawLandmarksAndSkeleton으로 통합
  - 불필요한 videoPreview/canvasPreview 등 코드 완전 제거

---

- 실시간 웹캠 영상(왼쪽)만 크게 표시, 비율 유지
- 관절 좌표 및 스켈레톤, 중심점, 척추선 실시간 시각화
- FPS(실제 렌더링 속도) 표시
- 분석 주기는 1초(1000ms)로 유지, landmarks는 최신 결과만 반영
- 코드 내 [AI 수정]/[AI 추가] 주석으로 변경 내역 명확히 표시

> 각 코드 변경에는 위와 같이 시간 주석(`// 2025-06-25 15:00 변경`)을 실제 코드에 추가해 추적할 수 있습니다. 