# Flaskbook 프로젝트 디렉토리 구조

## 개요
Flaskbook은 Flask 기반의 자세 분석 웹 애플리케이션입니다. MediaPipe를 활용하여 실시간으로 사용자의 자세를 분석하고 피드백을 제공합니다.

## 디렉토리 구조

```
flaskbook/
├── requirements.txt          # 프로젝트 의존성 파일
└── apps/                     # 메인 애플리케이션 디렉토리
    ├── app.py               # Flask 애플리케이션 팩토리 및 설정
    ├── __pycache__/         # Python 캐시 파일
    └── crud/                # 자세 분석 모듈
        ├── __init__.py      # Blueprint 초기화 (현재 미사용)
        ├── models.py        # 데이터베이스 모델 (User 클래스)
        ├── views.py         # 라우트 및 API 엔드포인트
        ├── neck.py          # 자세 분석 알고리즘
        ├── __pycache__/     # Python 캐시 파일
        ├── static/          # 정적 파일
        │   ├── style.css    # CSS 스타일 (미사용)
        │   └── crud/        # CRUD 관련 정적 파일
        │       └── script.js # 웹캠 처리 및 분석 요청 JavaScript
        └── templates/       # HTML 템플릿
            └── crud/        # CRUD 관련 템플릿
                └── index.html # 메인 분석 페이지
```

## 주요 컴포넌트 설명

### 1. 애플리케이션 설정 (`apps/app.py`)
- Flask 애플리케이션 팩토리 패턴 구현
- SQLAlchemy 데이터베이스 설정
- Blueprint 등록 및 라우팅 설정

### 2. 자세 분석 모듈 (`apps/crud/`)
- **views.py**: 웹캠 프레임 분석 API 엔드포인트
- **neck.py**: MediaPipe 기반 자세 분석 알고리즘
- **models.py**: 사용자 데이터 모델 (현재 미사용)

### 3. 프론트엔드 (`apps/crud/templates/`, `apps/crud/static/`)
- **index.html**: 실시간 자세 분석 웹 인터페이스
- **script.js**: 웹캠 제어 및 서버 통신 로직

## 기술 스택

### 백엔드
- **Flask**: 웹 프레임워크
- **Flask-SQLAlchemy**: ORM
- **MediaPipe**: 자세 인식 및 분석
- **OpenCV**: 이미지 처리
- **NumPy**: 수치 계산

### 프론트엔드
- **HTML5**: 웹 인터페이스
- **JavaScript**: 웹캠 제어 및 API 통신
- **CSS**: 스타일링 (최소한의 구현)

## 데이터베이스
- **SQLite**: 로컬 데이터베이스 (`local.sqlite`)
- **Flask-Migrate**: 데이터베이스 마이그레이션

## 주요 기능
1. 실시간 웹캠 자세 분석
2. 목 자세 (거북목) 분석
3. 척추 곡률 분석
4. 어깨 비대칭 분석
5. 골반 기울기 분석
6. 척추 틀어짐 분석 