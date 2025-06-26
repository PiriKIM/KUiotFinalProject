# CHANGELOG

## 2025-06-26

### 15:00
- User 모델에 password_hash 필드 및 비밀번호 해싱/검증 기능 추가 (`apps/crud/models.py`)
  ```python
  # 2024-07-05 15:00 변경
  password_hash = db.Column(db.String(255), nullable=False)
  @property
  def password(self):
      raise AttributeError("읽어 들일 수 없음")
  @password.setter
  def password(self, password):
      self.password_hash = generate_password_hash(password)
  def check_password(self, password):
      return check_password_hash(self.password_hash, password)
  ```

- PostureRecord 모델 및 종합 점수/등급 계산 기능 구현 (`apps/crud/models.py`)
  ```python
  # 2025-06-26 15:00 변경
  class PostureRecord(db.Model):
      ...
      def calculate_overall_score(self):
          ...
      def calculate_overall_grade(self):
          ...
  ```

### 15:10
- 회원가입/로그인/로그아웃/프로필 라우트 및 템플릿 추가 (`apps/crud/auth.py`, `templates/crud/`)
  - `/register`, `/login`, `/logout`, `/profile` 라우트 구현
  - 회원가입 시 중복 검사, 비밀번호 해싱, 플래시 메시지 등 추가
  - 프로필 페이지에서 최근 분석 기록, 통계 표시

### 15:20
- 자세 분석 결과 DB 저장 및 기록/통계/히스토리 기능 구현 (`apps/crud/views.py`)
  - `/analyze` 라우트에서 MediaPipe로 분석 후 PostureRecord에 저장
  - `/history` 라우트에서 사용자별 분석 기록 페이지네이션 제공
  - `/statistics` 라우트에서 통계(평균점수, 등급별 개수, 월별 통계 등) 제공

### 15:30
- 프론트엔드 템플릿 및 네비게이션 추가 (`templates/crud/`)
  - `index.html`, `login.html`, `register.html`, `profile.html`, `history.html`, `statistics.html` 등 구현
  - 실시간 분석, 기록/통계/프로필 UI 제공

### 15:40
- run.py 삭제됨 (서버 실행 스크립트)

---

### 2025-6-26

### 09:00
- html 파일 기존에 묶여있던 style css 코드 따로 css 파일 만들어서 분리
  -`apps/crud/static/analysis.css`
  -`apps/crud/static/auth.css`
  -`apps/crud/static/history.css`
  -`apps/crud/static/profile.css`
  -`apps/crud/static/statistics.css`
  -`apps/crud/static/style.css`


> 각 코드 변경에는 위와 같이 시간 주석(`// 2025-06-25 15:00 변경`)을 실제 코드에 추가해 추적할 수 있습니다. 