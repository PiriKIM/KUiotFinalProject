# 자세요정 API 문서

## 개요

자세요정 API는 AI 기반 체형 분석 및 자세 교정을 위한 RESTful API입니다.

- **Base URL**: `http://localhost:8000`
- **API 문서**: `http://localhost:8000/docs` (Swagger UI)
- **인증 방식**: JWT Bearer Token
- **데이터 형식**: JSON

## 인증

### JWT 토큰 사용법

```http
Authorization: Bearer <your-jwt-token>
```

## 엔드포인트 목록

### 1. 인증 (Authentication)

#### POST /auth/register
사용자 회원가입

**요청 본문:**
```json
{
  "username": "string",
  "email": "user@example.com",
  "password": "string",
  "full_name": "string"
}
```

**응답:**
```json
{
  "id": 1,
  "username": "string",
  "email": "user@example.com",
  "full_name": "string",
  "created_at": "2025-06-25T10:00:00Z"
}
```

#### POST /auth/login
사용자 로그인

**요청 본문:**
```json
{
  "username": "string",
  "password": "string"
}
```

**응답:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "string",
    "email": "user@example.com",
    "full_name": "string"
  }
}
```

#### GET /auth/me
현재 사용자 정보 조회

**헤더:**
```http
Authorization: Bearer <token>
```

**응답:**
```json
{
  "id": 1,
  "username": "string",
  "email": "user@example.com",
  "full_name": "string",
  "created_at": "2025-06-25T10:00:00Z"
}
```

### 2. 체형 분석 (Body Analysis)

#### POST /analyze/body
체형 분석 수행

**헤더:**
```http
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**요청 본문:**
```form-data
image: <image_file>
analysis_type: "posture" | "body_shape" | "both"
```

**응답:**
```json
{
  "analysis_id": "uuid",
  "user_id": 1,
  "analysis_type": "posture",
  "image_url": "string",
  "results": {
    "posture_score": 85.5,
    "body_angles": {
      "neck_angle": 15.2,
      "shoulder_angle": 2.1,
      "back_angle": 5.8
    },
    "recommendations": [
      "목을 더 세우세요",
      "어깨를 펴세요"
    ]
  },
  "created_at": "2025-06-25T10:00:00Z"
}
```

#### GET /analyze/body/{analysis_id}
특정 분석 결과 조회

**헤더:**
```http
Authorization: Bearer <token>
```

**응답:**
```json
{
  "analysis_id": "uuid",
  "user_id": 1,
  "analysis_type": "posture",
  "image_url": "string",
  "results": {
    "posture_score": 85.5,
    "body_angles": {
      "neck_angle": 15.2,
      "shoulder_angle": 2.1,
      "back_angle": 5.8
    },
    "recommendations": [
      "목을 더 세우세요",
      "어깨를 펴세요"
    ]
  },
  "created_at": "2025-06-25T10:00:00Z"
}
```

#### GET /analyze/body
사용자의 모든 분석 결과 조회

**헤더:**
```http
Authorization: Bearer <token>
```

**쿼리 파라미터:**
- `limit`: int (기본값: 10)
- `offset`: int (기본값: 0)
- `analysis_type`: string (선택사항)

**응답:**
```json
{
  "analyses": [
    {
      "analysis_id": "uuid",
      "analysis_type": "posture",
      "image_url": "string",
      "results": {
        "posture_score": 85.5
      },
      "created_at": "2025-06-25T10:00:00Z"
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

### 3. 실시간 자세 교정 (Real-time Posture Correction)

#### WebSocket /ws/posture-correction
실시간 자세 교정 웹소켓 연결

**연결:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/posture-correction');
```

**인증:**
```javascript
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));
};
```

**메시지 형식:**

클라이언트 → 서버:
```json
{
  "type": "frame",
  "data": "base64_encoded_image"
}
```

서버 → 클라이언트:
```json
{
  "type": "posture_feedback",
  "data": {
    "posture_score": 85.5,
    "current_angles": {
      "neck_angle": 15.2,
      "shoulder_angle": 2.1,
      "back_angle": 5.8
    },
    "feedback": "목을 더 세우세요",
    "is_good_posture": false
  }
}
```

### 4. 통계 (Statistics)

#### GET /statistics/overview
사용자 통계 개요

**헤더:**
```http
Authorization: Bearer <token>
```

**응답:**
```json
{
  "total_analyses": 25,
  "average_posture_score": 82.3,
  "best_posture_score": 95.0,
  "improvement_rate": 15.2,
  "analysis_count_by_month": [
    {
      "month": "2025-06",
      "count": 8
    }
  ]
}
```

#### GET /statistics/trends
자세 점수 트렌드

**헤더:**
```http
Authorization: Bearer <token>
```

**쿼리 파라미터:**
- `period`: string (기본값: "30d") - "7d", "30d", "90d", "1y"

**응답:**
```json
{
  "trends": [
    {
      "date": "2025-06-25",
      "average_score": 82.5,
      "analysis_count": 3
    }
  ],
  "improvement": {
    "overall": 15.2,
    "weekly": 2.1,
    "monthly": 8.5
  }
}
```

#### GET /statistics/body-parts
신체 부위별 분석 통계

**헤더:**
```http
Authorization: Bearer <token>
```

**응답:**
```json
{
  "body_parts": {
    "neck": {
      "average_angle": 12.5,
      "improvement": 8.2,
      "issues_count": 15
    },
    "shoulders": {
      "average_angle": 3.2,
      "improvement": 12.1,
      "issues_count": 8
    },
    "back": {
      "average_angle": 6.8,
      "improvement": 5.5,
      "issues_count": 22
    }
  }
}
```

## 에러 응답

### 일반적인 에러 코드

#### 400 Bad Request
```json
{
  "detail": "잘못된 요청입니다.",
  "errors": [
    {
      "field": "email",
      "message": "유효한 이메일 주소를 입력해주세요."
    }
  ]
}
```

#### 401 Unauthorized
```json
{
  "detail": "인증이 필요합니다."
}
```

#### 403 Forbidden
```json
{
  "detail": "접근 권한이 없습니다."
}
```

#### 404 Not Found
```json
{
  "detail": "요청한 리소스를 찾을 수 없습니다."
}
```

#### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "유효한 이메일 주소를 입력해주세요.",
      "type": "value_error.email"
    }
  ]
}
```

#### 500 Internal Server Error
```json
{
  "detail": "서버 내부 오류가 발생했습니다."
}
```

## 요청 제한

- **일반 API**: 분당 100회
- **분석 API**: 분당 10회
- **웹소켓 연결**: 사용자당 1개

## 데이터 형식

### 날짜/시간
ISO 8601 형식 사용: `YYYY-MM-DDTHH:MM:SSZ`

### 이미지
- **지원 형식**: JPEG, PNG, WebP
- **최대 크기**: 10MB
- **권장 해상도**: 1920x1080 이상

### 각도 측정
- **단위**: 도(degree)
- **범위**: 0-180도
- **정밀도**: 소수점 첫째 자리

## SDK 및 라이브러리

### JavaScript/TypeScript
```bash
npm install @jaseyojeong/api-client
```

### Python
```bash
pip install jaseyojeong-api-client
```

## 예제 코드

### JavaScript (Fetch API)
```javascript
// 로그인
const loginResponse = await fetch('/auth/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    username: 'user',
    password: 'password'
  })
});

const { access_token } = await loginResponse.json();

// 체형 분석
const formData = new FormData();
formData.append('image', imageFile);
formData.append('analysis_type', 'posture');

const analysisResponse = await fetch('/analyze/body', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`
  },
  body: formData
});
```

### Python (requests)
```python
import requests

# 로그인
login_data = {
    "username": "user",
    "password": "password"
}
response = requests.post("http://localhost:8000/auth/login", json=login_data)
token = response.json()["access_token"]

# 체형 분석
headers = {"Authorization": f"Bearer {token}"}
files = {"image": open("image.jpg", "rb")}
data = {"analysis_type": "posture"}

response = requests.post(
    "http://localhost:8000/analyze/body",
    headers=headers,
    files=files,
    data=data
)
```

## 지원

API 관련 문의사항이 있으시면 다음을 확인해주세요:
1. [설치 가이드](installation.md)
2. [개발 로그](development_log.md)
3. Swagger UI: http://localhost:8000/docs 