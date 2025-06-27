## [0.3.1] - 2025-06-27 00:30

### ✨ Added
- **실시간 스켈레톤 및 랜드마크 시각화 시스템**
  - `flaskbook/apps/crud/views.py`에 상체 랜드마크 데이터 전송 기능 추가 (얼굴 제외, 귀부터 시작)
  - `flaskbook/apps/crud/static/crud/script.js`에 실시간 스켈레톤 시각화 구현
    - 라임색 스켈레톤 연결선 (어깨, 팔, 다리, 목)
    - 빨간색 랜드마크 점들
    - 중간점 계산 및 표시 (목, 어깨, 골반, 척추)
    - 딥스카이블루 척추 중심선 (목-어깨-골반)
    - FPS 측정 및 표시 기능
- **Canvas 오버레이 시스템**
  - `flaskbook/apps/crud/templates/crud/index.html`에 Canvas 오버레이 추가
  - 비디오 위에 스켈레톤 오버레이 표시
  - FPS 표시 요소 추가
  - 실시간 시각적 피드백

### 🎨 UI/UX Improvements
- **실시간 스켈레톤 시각화**
  - 색상별 중간점 표시:
    - 🔵 파란색: 목 중심점
    - 🟠 주황색: 어깨 중심점
    - 🟣 보라색: 골반 중심점
    - 🔵 청록색: 척추 중심점
  - 딥스카이블루 척추 중심선 (목-어깨-골반)

### 🔄 System Architecture
- **실시간 시각화**: Canvas API + MediaPipe Pose 랜드마크

### 📂 Modified Files
- `flaskbook/apps/crud/views.py`: 랜드마크 데이터 전송 기능 추가
- `flaskbook/apps/crud/static/crud/script.js`: 스켈레톤 시각화 시스템 추가
- `flaskbook/apps/crud/templates/crud/index.html`: Canvas 오버레이 시스템 추가

---

## [0.3.0] - 2025-06-26 23:00

### 🔧 Changed
- `flaskbook/apps/crud/neck.py` 자세 분석 알고리즘 완전 업데이트
  - `side_pose_test.py`의 세밀한 분석 기준 적용
  - 거북목 분석에 `analyze_turtle_neck()` 메서드 추가
  - 모든 분석 메서드의 임계값 조정 및 코드 품질 개선
- `flaskbook/apps/crud/views.py` 상태 관리 시스템 도입
  - `PoseStateManager` 클래스 추가로 4단계 상태 관리 구현
  - 정면/측면 판별 로직 및 조건부 분석 시스템 적용
  - 상태별 JSON 응답 구조 개선
- `flaskbook/apps/crud/static/crud/script.js` 프론트엔드 상태 관리 구현
  - 실시간 상태 표시 및 색상 코딩 시스템
  - 진행률 바 및 안정화 시간 표시
  - 단계별 사용자 안내 메시지 시스템

### ✨ Added
- `flaskbook/apps/crud/utils/korean_font.py` 한글 폰트 렌더링 유틸리티
  - PIL 기반 한글 폰트 지원 시스템
  - 다중 폰트 경로 지원 및 폰트 캐싱
  - 오류 처리 및 fallback 메커니즘
- `flaskbook/requirements.txt` 의존성 추가
  - `Flask-Migrate>=4.0`: 데이터베이스 마이그레이션
  - `Pillow>=9.0.0`: 한글 폰트 렌더링 지원
- `flaskbook/apps/crud/templates/crud/index.html` 현대적 UI 디자인
  - 글래스모피즘 효과 및 그라데이션 배경
  - 상태별 시각적 피드백 시스템
  - 반응형 디자인 및 모바일 지원
  - 실시간 진행률 표시 및 단계별 안내

### 🎨 UI/UX Improvements
- 상태별 색상 코딩 시스템 구현
  - `no_human_detected`: 빨간색 (사람 미감지)
  - `detecting_front_pose`: 청록색 (정면 자세 감지 중)
  - `waiting_side_pose`: 파란색 (측면 자세 대기 중)
  - `analyzing_side_pose`: 초록색 (자세 분석 중)
- 실시간 진행률 바 및 안정화 시간 표시
- 이모지 아이콘을 활용한 직관적인 분석 결과 표시
- 동적 버튼 텍스트 및 상태별 메시지 시스템
- **완전한 반응형 디자인 구현**
  - 모바일, 태블릿, 데스크톱 모든 화면 크기 지원
  - CSS Grid와 Flexbox를 활용한 유연한 레이아웃
  - 뷰포트 기반 동적 크기 조정 (vw, vh, rem 단위 활용)
  - 미디어 쿼리를 통한 브레이크포인트별 최적화
    - 모바일: 768px 이하 (세로 레이아웃, 큰 터치 버튼)
    - 태블릿: 768px-1024px (중간 크기 레이아웃)
    - 데스크톱: 1024px 이상 (가로 레이아웃, 세밀한 UI)
- **접근성 개선**
  - 터치 친화적 버튼 크기 (최소 44px)
  - 고대비 색상 조합으로 가독성 향상
  - 키보드 네비게이션 지원
  - 스크린 리더 호환성 고려

### 🔄 System Architecture
- 백엔드: Flask + MediaPipe + SQLAlchemy 기반 상태 관리
- 프론트엔드: JavaScript + HTML5 Canvas + WebRTC
- 데이터베이스: SQLite + 사용자별 분석 기록 저장
- 한글 지원: PIL 기반 폰트 렌더링 시스템

### 📂 Modified Files
- `flaskbook/apps/crud/neck.py`: 자세 분석 알고리즘 업데이트
- `flaskbook/apps/crud/views.py`: 상태 관리 시스템 추가
- `flaskbook/apps/crud/static/crud/script.js`: 프론트엔드 상태 관리
- `flaskbook/apps/crud/templates/crud/index.html`: UI/UX 개선
- `flaskbook/requirements.txt`: 의존성 추가
- `flaskbook/apps/crud/utils/korean_font.py`: 새로 생성
- `flaskbook/apps/crud/utils/__init__.py`: 새로 생성

---

## [0.2.0] - 2025-06-26 16:00

### 🔧 Changed
- 자세 상태 관리 클래스를 완전히 재설계하여 다음 4단계 상태로 변경:
  - `no_human_detected`, `detecting_front_pose`, `waiting_side_pose`, `analyzing_side_pose`
  - 파일: `PoseStateManager` class in `test/side_pose_test.py`
- 정면 안정화 로직 구현 (landmark 이동량 평균 기반)
  - 평균 20프레임 누적 후 어깨-귀 사각형 면적 계산
  - 파일: `PoseStateManager` in `test/side_pose_test.py`
- 측면 판별 기준 변경 (정면 대비 면적 70% 이하)
  - 파일: `PoseStateManager`, main loop in `test/side_pose_test.py`
- 한글 텍스트 렌더링 시스템 도입
  - PIL 기반 한글 폰트 지원으로 상태 메시지 한글화
  - 파일: `put_korean_text()`, `get_korean_font()` in `test/side_pose_test.py`

### 🐛 Fixed
- 상태 전이 오류로 인한 메시지 반복 출력 현상 수정
  - 관련 위치: `test/side_pose_test.py`, `get_state_message()`
- 정면 자세 감지 불안정 문제 해결
  - 유예시간 로직 추가로 일시적 조건 변화 시 타이머 유지
  - 파일: `PoseStateManager.update_state()` in `test/side_pose_test.py`

### ✨ Added
- 실시간 상태 표시 기능
  - 화면 상단에 현재 상태 메시지 한글 표시
  - 정면 자세 대기 중 안정 시간 표시
  - 파일: main loop in `test/side_pose_test.py`
- 측면 자세 분석 시에만 자세 분석 실행
  - `analyzing_side_pose` 상태에서만 5가지 자세 분석 수행
  - 파일: main loop in `test/side_pose_test.py`

### 📂 관련 파일
- `test/side_pose_test.py` (기존 `yj/Back_End/MediaPipe_test/webcam_pose_test.py`에서 리팩토링)
- `PoseStateManager` 클래스 내 상태 관리 로직
- `get_state_message()`, `update_state()`, 정면·측면 판별 부분
- 한글 텍스트 렌더링 함수들

---