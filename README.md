# 자세요정 (AI 기반 체형 분석 및 자세 교정 시스템)

## 📋 프로젝트 개요

**자세요정**은 AI 기술을 활용하여 현대인의 체형을 분석하고 실시간으로 자세를 교정해주는 웹 애플리케이션입니다. 

### 🎯 주요 기능
- **체형 분석 모드**: 웹캠/사진을 통한 전신 체형 분석 및 개선 권장사항 제공
- **실시간 자세 교정 모드**: 웹캠을 통한 실시간 자세 모니터링 및 TTS 알림
- **데이터 관리**: 사용자별 체형 변화 추이 및 통계 제공

### 🏥 타겟 사용자
- 장시간 앉아서 업무/학습하는 직장인 및 학생
- 체형 교정에 관심이 있는 일반인
- 만성적인 목/어깨 통증을 겪는 사람들

## 🛠 기술 스택

### 백엔드
- **언어**: Python 3.9+
- **웹 프레임워크**: FastAPI
- **AI/Computer Vision**: OpenCV, MediaPipe, NumPy
- **데이터베이스**: SQLite (개발) / PostgreSQL (운영)
- **인증**: JWT
- **TTS**: pyttsx3 또는 gTTS

### 프론트엔드
- **언어**: HTML5, CSS3, JavaScript
- **프레임워크**: React.js
- **UI 라이브러리**: Bootstrap 또는 Tailwind CSS
- **차트**: Chart.js 또는 D3.js

## 🚀 빠른 시작

### 사전 요구사항
- Python 3.9+
- Node.js 16+
- 웹캠 지원 디바이스

### 설치 및 실행

1. **저장소 클론**
```bash
git clone https://github.com/your-username/jaseyojeong.git
cd jaseyojeong
```

2. **백엔드 설정**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **프론트엔드 설정**
```bash
cd frontend
npm install
```

4. **데이터베이스 설정**
```bash
cd backend
alembic upgrade head
```

5. **서버 실행**
```bash
# 백엔드 (터미널 1)
cd backend
uvicorn app.main:app --reload

# 프론트엔드 (터미널 2)
cd frontend
npm start
```

6. **브라우저에서 접속**
- 프론트엔드: http://localhost:3000
- 백엔드 API: http://localhost:8000
- API 문서: http://localhost:8000/docs

## 📁 프로젝트 구조

```
FinalProject/
├── backend/                 # FastAPI 백엔드
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI 앱 진입점
│   │   ├── models/         # Pydantic 모델
│   │   ├── routers/        # API 라우터
│   │   ├── services/       # 비즈니스 로직
│   │   ├── utils/          # 유틸리티 함수
│   │   └── database/       # 데이터베이스 설정
│   ├── requirements.txt    # Python 의존성
│   └── alembic/           # 데이터베이스 마이그레이션
├── frontend/               # React 프론트엔드
│   ├── public/
│   ├── src/
│   │   ├── components/     # React 컴포넌트
│   │   ├── pages/         # 페이지 컴포넌트
│   │   ├── services/      # API 서비스
│   │   └── utils/         # 유틸리티 함수
│   ├── package.json
│   └── README.md
├── docs/                   # 프로젝트 문서
├── tests/                  # 테스트 파일
├── PRD.md                  # 제품 요구사항 문서
├── .cursorrules           # 커서 에이전트 규칙
└── README.md              # 프로젝트 README
```

## 🔧 주요 기능 상세

### 1. 체형 분석 모드
- **전신 촬영**: 웹캠 또는 사진 업로드를 통한 전신 분석
- **정면/측면 분석**: 척추 정렬, 어깨 높이, 골반 기울기 등 분석
- **개선 권장사항**: 분석 결과에 따른 맞춤형 운동 및 자세 교정 가이드

### 2. 실시간 자세 교정 모드
- **실시간 모니터링**: 웹캠을 통한 지속적인 자세 감지
- **각도 계산**: 귀, 턱, 목, 어깨, 골반 좌표를 이용한 거북목 각도 측정
- **등급 시스템**: A/B/C/D 등급으로 자세 상태 분류
- **TTS 알림**: 나쁜 자세 지속 시 음성 알림

### 3. 데이터 관리
- **변화 추이**: 체형 변화를 그래프로 시각화
- **통계 분석**: 자세 개선 통계 및 리포트
- **이력 관리**: 이전 분석 데이터 저장 및 비교

## 🔌 API 엔드포인트

### 인증
- `POST /auth/register` - 회원가입
- `POST /auth/login` - 로그인
- `POST /auth/refresh` - 토큰 갱신

### 체형 분석
- `POST /analyze/body` - 체형 분석
- `GET /analyze/body/{user_id}` - 사용자 체형 분석 이력

### 자세 교정
- `POST /analyze/posture` - 자세 분석
- `WebSocket /ws/posture` - 실시간 자세 모니터링

### 통계
- `GET /statistics/{user_id}` - 사용자 통계
- `GET /statistics/{user_id}/trend` - 변화 추이

## 🧪 테스트

### 백엔드 테스트
```bash
cd backend
pytest
```

### 프론트엔드 테스트
```bash
cd frontend
npm test
```

### E2E 테스트
```bash
npm run cypress:open
```

## 📊 성능 지표

- **자세 감지 정확도**: > 90%
- **실시간 처리 지연**: < 100ms
- **API 응답 시간**: < 2초
- **체형 분석 시간**: < 5초

## 🔒 보안

- **JWT 기반 인증** 시스템
- **개인정보 암호화** 저장
- **HTTPS** 통신 필수
- **API 요청 제한** 구현

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.

## 🙏 감사의 말

- [MediaPipe](https://mediapipe.dev/) - AI 포즈 감지 기술
- [OpenCV](https://opencv.org/) - 컴퓨터 비전 라이브러리
- [FastAPI](https://fastapi.tiangolo.com/) - 현대적인 웹 프레임워크
- [React](https://reactjs.org/) - 사용자 인터페이스 라이브러리

---

**자세요정**으로 건강한 자세를 만들어보세요! 🧘‍♀️✨ 