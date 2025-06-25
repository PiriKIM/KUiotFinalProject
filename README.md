# FlaskBook 자세 분석 시스템 (Posture Analysis Web App)

## 소개 (Introduction)
이 프로젝트는 Flask, MediaPipe, OpenCV, SQLAlchemy를 활용한 **실시간 자세 분석 웹앱**입니다. 회원가입/로그인, 실시간 분석, 기록/통계/프로필 등 다양한 기능을 제공합니다.

This project is a real-time posture analysis web app using Flask, MediaPipe, OpenCV, and SQLAlchemy. It supports user registration/login, real-time analysis, history/statistics/profile, and more.

## 주요 기능 (Key Features)
- 회원가입/로그인/로그아웃 (User registration/login/logout)
- 실시간 웹캠 자세 분석 (Real-time webcam posture analysis)
- 분석 결과 DB 저장 및 사용자별 기록 관리 (DB storage & per-user history)
- 기록/통계/프로필 페이지 (History/statistics/profile pages)
- 반응형 UI (Responsive UI)
- 등급별/월별/트렌드 차트 (Grade/month/trend charts)

## 폴더 구조 (Folder Structure)
- `apps/crud/` : 주요 뷰, 모델, 분석로직 (Main views, models, analysis logic)
- `apps/templates/crud/` : 모든 HTML 템플릿 (All HTML templates)
- `apps/app.py` : Flask 앱 팩토리 (Flask app factory)
- `init_db.py` : DB 초기화 (DB initialization)
- `create_test_data.py` : 테스트 데이터 생성 (Test data generation)
- `run.py` : 앱 실행 스크립트 (App runner)

## 설치 및 실행 (Setup & Run)
```bash
pip install -r requirements.txt
python3 init_db.py
python3 create_test_data.py   # (선택 Optional)
python3 run.py
```

## 접속 (Access)
- http://localhost:5000/auth/login

## 테스트 계정 (Test Account)
- ID: test_user
- PW: password123

## 기술스택 (Tech Stack)
- Python, Flask, SQLAlchemy, MediaPipe, OpenCV, Jinja2, HTML/CSS/JS

## 문의 (Contact)
- 담당자: [이름/이메일]
