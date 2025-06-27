# 설치 및 실행 안내

## 1. 의존성 설치
```bash
pip install -r requirements.txt
```

## 2. 데이터베이스 초기화
```bash
python3 init_db.py
```

## 3. (선택) 테스트 데이터 생성
```bash
python3 create_test_data.py
```

## 4. 서버 실행
```bash
python3 run.py
```

## 5. 웹 접속
- http://localhost:5000/auth/login

## 6. 테스트 계정
- ID: test_user
- PW: password123

## 7. 주요 기능
- 회원가입/로그인/로그아웃
- 실시간 자세 분석 및 결과 저장
- 기록/통계/프로필 페이지
- 반응형 UI

## 8. 기타
- DB 파일: `flaskbook/local.sqlite`
- 설정 변경: `flaskbook/apps/app.py` 참고 