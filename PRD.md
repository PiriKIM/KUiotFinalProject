
## 주요 파일 설명

- **data/image.png**  
  바른자세 관절 좌표 수치 추출용 이미지 데이터 폴더

- **doc/WORKFLOW.md, 플로우차트.png**  
  시스템 전체 흐름, 데이터 플로우, 기술 스택, 확장 아이디어 등 문서화

- **Back_End/MediaPipe_test/**  
  - `webcam_pose_test.py`: MediaPipe로 실시간 자세 분석(기본)
  - `webcam_pose_test_neck.py`: MediaPipe로 거북목 각도 측정 및 등급화, 실시간 피드백
  - `analyze_image_pose.py`: 이미지에서 목 각도 추출(등급 기준 설정용)
  - 기타 MediaPipe 기반 분석/실험 코드
<!-- 
- **OpenPose/**  
  - `webcam_pose_test.py`: OpenPose로 실시간 자세 분석
  - `person_tracker.py`: 마우스로 특정 인물 선택 후 추적
  - `README.md`, `INSTALL.md`: 설치 및 사용법 문서 -->
- **Front_End/**  
  - 웹 페이지 개발용 코드 폴더
---

## 2. 제품 요구 명세서 (`PRD.md`)

```markdown
# Product Requirements Document (PRD)
## 1. 목적
- 사용자가 웹캠을 통해 실시간으로 자신의 자세(특히 거북목 등)를 분석하고, 시각적/텍스트 피드백을 즉시 제공받을 수 있는 웹 서비스 제공

## 2. 주요 기능
### 2.1 사용자 기능
- 웹사이트 회원가입/로그인
- 웹캠 접근 및 실시간 영상 스트림 전송
- 실시간 자세 분석 결과(등급, 피드백) 확인
- 자세 교정 가이드 및 시각화(관절 오버레이 등) 제공

### 2.2 서버/분석 기능
- 영상 프레임 수신 및 저장
- OpenPose/MediaPipe 기반 관절 추출
- 거북목, 척추, 어깨, 골반 등 주요 자세 이슈 분석
- 거북목 각도 등급화(A/B/C/D) 및 실시간 피드백 생성
- 관절 좌표, 분석 결과, 피드백 데이터(JSON) 생성 및 전송

### 2.3 확장 기능(선택)
- 특정 인물 선택 및 추적(마우스 ROI)
- 자세 이력 저장 및 리포트
- 다양한 운동/자세별 분석 지원

## 3. 시스템 워크플로우
- [사용자 웹캠] → [프론트엔드] → [서버] → [OpenPose/MediaPipe 분석] → [피드백 생성] → [프론트엔드] → [사용자]
- 상세 단계 및 다이어그램: `doc/WORKFLOW.md`, `doc/플로우차트.png` 참고

## 4. 기술 스택
- 프론트엔드: React.js, WebRTC/WebSocket, HTML5 Video
- 백엔드: Python (Flask/FastAPI), OpenPose, MediaPipe, WebSocket/REST API
- 서버: GPU 지원 Linux 서버, Docker
- 기타: HTTPS, JWT 인증, 로그/모니터링

## 5. 성능 및 품질 요구사항
- 실시간 분석(지연 1초 이내)
- 다양한 해상도/환경 지원(PC, VM 등)
- 사용자 친화적 UI/UX, 명확한 피드백 제공
- 보안(인증, 데이터 암호화 등)

## 6. 기타 참고
- 자세한 워크플로우 및 시스템 구조는 `doc/WORKFLOW.md` 및 플로우차트 이미지 참고
- 코드 구조 및 각 파일 역할은 `DIRECTORY.md` 참고
```

