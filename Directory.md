# 프로젝트 디렉토리 구조 

KUiotFinalProject/
│
├── data/
│ ├── image.png # 바른자세 수치 추출용 데이터 이미지
│ 
├── doc/
│ ├── WORKFLOW.md # 전체 시스템 워크플로우 및 플로우차트 설명
│ └── 플로우차트.png # 시스템 플로우차트 이미지
│
├── yj/
│ └── Back_End/
│ ├── MediaPipe_test/
│ │ ├── webcam_pose_test.py # MediaPipe 기반 실시간 자세 분석(기본)
│ │ ├── webcam_pose_test_neck.py # MediaPipe 기반 거북목 분석 및 등급화
│ │ ├── analyze_image_pose.py # 이미지 기반 자세 분석(거북목 기준 각도 추출)
│ │ └── ... # 기타 MediaPipe 관련 스크립트
<!-- │ │
│ └── OpenPose/
│ ├── webcam_pose_test.py # OpenPose 기반 실시간 자세 분석
│ ├── person_tracker.py # OpenPose 기반 특정 인물 추적 기능
│ ├── README.md # OpenPose 사용법 및 설치 안내
│ ├── INSTALL.md # OpenPose 설치 상세 가이드
│ └── ... # 기타 OpenPose 관련 스크립트
│ -->
└── (기타 프론트엔드, 서버, 데이터, 설정 파일 등)