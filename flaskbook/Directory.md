# Flaskbook 프로젝트 디렉토리 구조

## 개요
Flaskbook은 Flask 기반의 자세 분석 웹 애플리케이션입니다. MediaPipe를 활용하여 실시간으로 사용자의 자세를 분석하고, 로그인 시스템을 통해 개인별 분석 결과를 데이터베이스에 저장하여 피드백을 제공합니다.

## 디렉토리 구조

```
flaskbook/
├── requirements.txt          # 프로젝트 의존성 파일
├── init_db.py               # 데이터베이스 초기화 스크립트
├── run.py                   # Flask 애플리케이션 실행 스크립트
└── apps/                    # 메인 애플리케이션 디렉토리
    ├── app.py              # Flask 애플리케이션 팩토리 및 설정
    ├── __pycache__/        # Python 캐시 파일
    └── crud/               # 자세 분석 모듈
        ├── __init__.py     # Blueprint 초기화 (현재 미사용)
        ├── models.py       # 데이터베이스 모델 (User, PostureRecord)
        ├── views.py        # 라우트 및 API 엔드포인트
        ├── auth.py         # 인증 관련 라우트 (로그인/회원가입)
        ├── neck.py         # 자세 분석 알고리즘
        ├── __pycache__/    # Python 캐시 파일
        ├── static/         # 정적 파일
        │   ├── style.css   # CSS 스타일 (미사용)
        │   └── crud/       # CRUD 관련 정적 파일
        │       └── script.js # 웹캠 처리 및 분석 요청 JavaScript
        └── templates/      # HTML 템플릿
            └── crud/       # CRUD 관련 템플릿
                ├── index.html    # 메인 분석 페이지
                ├── login.html    # 로그인 페이지
                ├── register.html # 회원가입 페이지
                ├── profile.html  # 사용자 프로필 페이지
                ├── history.html  # 분석 기록 페이지
                └── statistics.html # 통계 페이지
```

## 주요 컴포넌트 설명

### 1. 애플리케이션 설정 (`apps/app.py`)
- Flask 애플리케이션 팩토리 패턴 구현
- SQLAlchemy 데이터베이스 설정 (SQLite)
- Blueprint 등록 및 라우팅 설정 (crud, auth)
- 세션 관리 설정

### 2. 데이터베이스 모델 (`apps/crud/models.py`)
- **User 모델**: 사용자 정보, 로그인 시간, 비밀번호 해싱
- **PostureRecord 모델**: 자세 분석 결과 저장, 종합 점수 계산
- 관계 설정 (User ↔ PostureRecord)

### 3. 인증 시스템 (`apps/crud/auth.py`)
- **회원가입**: 사용자 등록 및 유효성 검사
- **로그인**: 세션 기반 인증
- **로그아웃**: 세션 정리
- **프로필**: 사용자 정보 및 분석 통계

### 4. 자세 분석 모듈 (`apps/crud/`)
- **views.py**: 웹캠 프레임 분석 API, 데이터베이스 저장, 기록/통계 페이지
- **neck.py**: MediaPipe 기반 자세 분석 알고리즘
- **로그인 데코레이터**: 인증 필요 페이지 보호

### 5. 프론트엔드 (`apps/crud/templates/`, `apps/crud/static/`)
- **index.html**: 실시간 자세 분석 웹 인터페이스 (네비게이션 포함)
- **login.html**: 로그인 페이지
- **script.js**: 웹캠 제어, 서버 통신, 종합 점수 표시

### 6. 실행 스크립트
- **init_db.py**: 데이터베이스 초기화 및 테이블 생성
- **run.py**: Flask 애플리케이션 실행

## 기술 스택

### 백엔드
- **Flask**: 웹 프레임워크
- **Flask-SQLAlchemy**: ORM
- **Flask-Migrate**: 데이터베이스 마이그레이션
- **MediaPipe**: 자세 인식 및 분석
- **OpenCV**: 이미지 처리
- **NumPy**: 수치 계산
- **Werkzeug**: 비밀번호 해싱

### 프론트엔드
- **HTML5**: 웹 인터페이스
- **CSS3**: 스타일링 (그라디언트, 반응형)
- **JavaScript**: 웹캠 제어 및 API 통신

## 데이터베이스
- **SQLite**: 로컬 데이터베이스 (`local.sqlite`)
- **User 테이블**: 사용자 정보, 로그인 시간
- **PostureRecord 테이블**: 자세 분석 결과, 종합 점수

## 주요 기능

### 1. 사용자 관리
- 회원가입 및 로그인
- 세션 기반 인증
- 사용자별 데이터 분리

### 2. 실시간 자세 분석
- 웹캠 기반 실시간 분석
- 목 자세 (거북목) 분석
- 척추 곡률 분석
- 어깨 비대칭 분석
- 골반 기울기 분석
- 척추 틀어짐 분석

### 3. 데이터 저장 및 관리
- 모든 분석 결과 데이터베이스 저장
- 종합 점수 및 등급 계산 (0-100점, A-D등급)
- 분석 시간 기록

### 4. 기록 및 통계
- 개인별 분석 기록 조회
- 통계 및 트렌드 분석
- 월별/주별 성과 추적

### 5. 사용자 인터페이스
- 네비게이션 바
- 종합 점수 실시간 표시
- 반응형 디자인
- 플래시 메시지 시스템 