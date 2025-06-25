# posture-web-analyzer 디렉토리 구조 설명

이 문서는 `umi` 브랜치에서 설계한 FastAPI 기반 자세 분석 웹 프로젝트의 디렉토리 구조를 설명합니다.  
본 구조는 **MediaPipe 기반 AI 자세 분석 기능을 FastAPI 서버로 제공**하고,  
웹 페이지에서 사용자의 웹캠 이미지를 받아 실시간 분석 피드백을 제공하기 위해 설계되었습니다.

---

## 📁 디렉토리 구조

```
posture-web-analyzer/
├── app/                        # 🔧 서버 애플리케이션 코드
│   ├── __init__.py             # 패키지 초기화 파일
│   ├── main.py                 # 🌐 FastAPI 서버 엔트리포인트
│   ├── analyzer/               # 🧠 자세 분석 및 유틸리티 모듈
│   │   ├── __init__.py
│   │   ├── posture.py          # MediaPipe 자세 분석 클래스 및 알고리즘
│   │   ├── utils.py            # 각도/거리 계산 등 유틸 함수
│   │   └── draw.py             # OpenCV 기반 시각화 함수
│   ├── camera/                 # 📷 입력 처리 모듈
│   │   ├── __init__.py
│   │   └── webcam.py           # 웹캠/비디오 입력 처리
│   ├── models/                 # 🧱 Pydantic 기반 API 입출력 모델 정의
│   │   ├── __init__.py
│   │   └── posture_result.py   # 분석 결과 반환 데이터 모델
│   └── config.py               # ⚙️ 설정 파일
│
├── static/                     # 🖼️ 정적 파일 (JS, CSS, 이미지 등)
│   ├── script.js               # 웹캠 캡처 및 분석 요청 JS
│   └── style.css               # 웹페이지 스타일
│
├── templates/                  # 🌐 Jinja2 기반 HTML 템플릿
│   └── index.html              # 웹캠 캡처 및 분석 UI
│
├── test/                       # ✅ 유닛 테스트 및 분석 모듈 테스트
│   └── test_analyzer.py        # 분석 클래스 테스트 코드
│
├── requirements.txt            # 📦 설치 패키지 목록
├── README.md                   # 📘 프로젝트 개요 및 실행 안내
└── run.sh                      # 🔁 서버 실행 스크립트 (uvicorn)
```

---

## 🎯 프로젝트 비전

- **누구나 쉽게 자세 분석**: 웹캠만 있으면 자세 상태를 바로 확인할 수 있도록 설계
- **확장 가능한 구조**: 분석 기능, 입력, 프론트엔드가 분리되어 다양한 UI/기능 확장 용이
- **헬스케어, 교육, 피트니스 등 다양한 환경 지원**: 운동 코칭, 학생 자세 모니터링 등으로 확장 가능

---

## 📌 설계 배경 및 특징

- **FastAPI 채택**  
  → 빠른 응답, 비동기 처리, 자동 문서화 등 현대적 API 서버에 최적화

- **app/ 하위 모듈 분리**  
  → 분석, 입력, 유틸, 모델 등 역할별로 코드 분리하여 유지보수와 테스트 용이성 확보

- **템플릿/정적 폴더 구분**  
  → FastAPI의 `Jinja2Templates`로 HTML 렌더링, JS로 웹캠 캡처 및 분석 요청 분리 구현

- **models/ 디렉토리 도입**  
  → API 입출력 데이터 타입 명확화 및 검증

---

## 🗂️ 사용 예시

### FastAPI 서버 실행 (개발 모드)

```bash
cd posture-web-analyzer
uvicorn app.main:app --reload
```

- 웹 브라우저에서 http://localhost:8000 접속 시 웹캠 기반 자세 분석 페이지 이용 가능

---

## 버전 기록 (Changelog)

| 날짜         | 작업 내역                                  |
| ------------ | ------------------------------------------ |
| 2025-06-25   | 디렉토리 구조 설계 및 모듈 분리 완료        |
| 2025-06-25   | FastAPI 서버 구성 및 분석 모듈 구조화 반영   |

📝 자세한 개발 흐름은 [dev_log.md](dev_log.md) 문서를 참고하세요.

---
트
- 브랜치 : umi