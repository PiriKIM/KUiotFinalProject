## [0.2.0] - 2025-06-26

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