# 자세요정 설치 가이드

## 시스템 요구사항

### 필수 요구사항
- **Python**: 3.9 이상
- **Node.js**: 18.0 이상
- **npm**: 8.0 이상
- **Git**: 최신 버전

### 권장 사양
- **RAM**: 8GB 이상
- **저장공간**: 10GB 이상
- **웹캠**: 실시간 자세 교정 기능 사용 시 필요

## 개발 환경 설정

### 1. 저장소 클론
```bash
git clone <repository-url>
cd FinalProject
```

### 2. 백엔드 설정

#### Python 가상환경 생성
```bash
# 백엔드 디렉토리로 이동
cd backend

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### 의존성 설치
```bash
# 기본 패키지 설치
pip install -r requirements.txt

# 추가 패키지 설치 (AI 엔진용)
pip install mediapipe opencv-python numpy pillow
pip install pyttsx3 gtts
pip install alembic psycopg2-binary
```

#### 환경변수 설정
```bash
# .env 파일 생성
cp env.example .env

# .env 파일 편집
nano .env
```

필요한 환경변수:
```env
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 3. 프론트엔드 설정

#### Node.js 패키지 설치
```bash
# 프론트엔드 디렉토리로 이동
cd ../frontend

# 의존성 설치
npm install
```

#### 환경변수 설정 (선택사항)
```bash
# .env 파일 생성
cp .env.example .env
```

## 데이터베이스 설정

### SQLite (개발용)
```bash
cd backend

# 데이터베이스 마이그레이션
alembic upgrade head
```

### PostgreSQL (운영용)
```bash
# PostgreSQL 설치 후
pip install psycopg2-binary

# 환경변수 수정
DATABASE_URL=postgresql://username:password@localhost:5432/jaseyojeong
```

## 실행 방법

### 1. 백엔드 서버 실행
```bash
cd backend

# 가상환경 활성화
source venv/bin/activate

# FastAPI 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 프론트엔드 개발 서버 실행
```bash
cd frontend

# React 개발 서버 실행
npm start
```

### 3. 접속 확인
- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **프론트엔드**: http://localhost:3000

## 개발 도구 설정

### VS Code 확장 프로그램 (권장)
- Python
- TypeScript and JavaScript Language Features
- React Developer Tools
- SQLite Viewer
- REST Client

### 디버깅 설정
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": ["app.main:app", "--reload"],
            "cwd": "${workspaceFolder}/backend"
        }
    ]
}
```

## 문제 해결

### 일반적인 문제들

#### 1. Python 패키지 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# 캐시 클리어
pip cache purge
```

#### 2. Node.js 패키지 설치 오류
```bash
# npm 캐시 클리어
npm cache clean --force

# node_modules 삭제 후 재설치
rm -rf node_modules package-lock.json
npm install
```

#### 3. 포트 충돌
```bash
# 포트 사용 확인
lsof -i :8000
lsof -i :3000

# 프로세스 종료
kill -9 <PID>
```

#### 4. 웹캠 접근 권한
```bash
# 브라우저에서 웹캠 권한 허용 필요
# Chrome: chrome://settings/content/camera
# Firefox: about:preferences#privacy
```

## 추가 설정

### Git Hooks 설정
```bash
# pre-commit 설치
pip install pre-commit

# pre-commit 설정
pre-commit install
```

### 코드 포맷팅
```bash
# 백엔드
pip install black isort flake8
black app/
isort app/
flake8 app/

# 프론트엔드
npm install --save-dev prettier eslint
npx prettier --write src/
```

## 지원

문제가 발생하면 다음을 확인해주세요:
1. 시스템 요구사항 충족 여부
2. 환경변수 설정 확인
3. 의존성 패키지 버전 호환성
4. [개발 로그](development_log.md) 참조

추가 도움이 필요하면 이슈를 등록해주세요. 