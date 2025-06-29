# 개발 일지 (Development Log)

## 향후 작업 계획

### 🎯 주요 목표
- 실시간 자세 분석 시스템 완성도 향상
- 사용자 경험 개선 및 안정성 확보
- 수집된 데이터를 활용한 머신러닝 모델 개발

### 📋 작업 항목

#### 1. 데이터 수집 및 분석 시스템 고도화
- [ ] **데이터 품질 개선**
  - 수집된 395개 레코드 데이터 품질 검증
  - 정면/측면 라벨링 정확도 검토 및 재라벨링
  - 노이즈 데이터 필터링 알고리즘 적용

- [ ] **다양한 사람 데이터 수집**
  - 현재 P1 데이터만 있으므로 P2, P3 등 추가 사람 데이터 수집
  - 체형별, 연령대별 데이터 다양성 확보
  - 각 사람별 정면/측면 데이터 균형 맞추기
  - **팀원 전체 테스트 계획**
    - 팀원 5명이 각각 P1, P2, P3, P4, P5 로 구분하여 데이터 수집
    - 각 팀원별로 정면/측면 자세 데이터 균형 있게 수집 (각각 50-100개씩)
    - 체형, 키, 자세 습관 등 개인차를 반영한 다양한 데이터 확보
    - 팀원별 테스트 일정 및 역할 분담 계획 수립

- [ ] **현재 v1 랜드마크 기반 분석 정밀도 향상**
  - 33개 랜드마크를 활용한 더 정교한 자세 분석 알고리즘 개발
  - 랜드마크 간 관계성 분석 강화
  - 자세 상태 판별 정확도 개선

#### 2. 머신러닝 모델 개발 및 학습
- [x] **정면/측면 분류 모델 구현**
  - 수집된 데이터를 활용한 지도학습 모델 개발
  - KNN, SVM, 랜덤포레스트 등 다양한 알고리즘 비교
  - 교차 검증을 통한 모델 성능 평가

- [x] **실시간 자세 분류 시스템**
  - 학습된 모델을 실시간 웹캠 스트림에 적용
  - 수동 라벨링 없이 자동으로 정면/측면 판별
  - 분류 결과 시각화 및 피드백 제공

- [x] **모델 성능 최적화**
  - 하이퍼파라미터 튜닝
  - 특성 선택 및 엔지니어링
  - 모델 앙상블 기법 적용

#### 3. 자세 분석 알고리즘 최적화
- [ ] **목 자세 분석 정확도 향상**
  - 현재 거북목 감지 임계값 조정
  - 목 각도 계산 방식 개선
  - 안정화 시간 조정 (현재 20프레임 → 최적값 찾기)

- [ ] **골반 자세 분석 추가**
  - 골반 기울기 감지
  - 골반 회전 분석
  - 골반-어깨 정렬 확인

- [ ] **척추 기울기 분석 추가**
  - 척추 중심선 기울기 측정
  - 척추 굽힘 정도 분석
  - 척추-골반 정렬 확인

#### 4. 프론트엔드 UI/UX 개선
- [ ] **캔버스 깜빡임 문제 해결**
  - Canvas 렌더링 최적화
  - 더블 버퍼링 적용
  - 프레임 스킵 로직 개선
  - GPU 가속 활용

- [ ] **점수 표시 대신 상태 알림 시스템**
  - 실시간 점수 표시 제거 (집중도 저하 방지)
  - 자세 상태별 간단한 알림 메시지
  - 문제가 있을 때만 알림 표시
  - 학습/업무 중 방해 최소화

- [ ] **분석 결과 시각화 개선**
  - 자세별 상세 점수 표시
  - 개선 방향 가이드 추가
  - 히스토리 차트 구현

- [ ] **사용자 피드백 시스템**
  - 자세 교정 운동 가이드
  - 실시간 피드백 메시지 개선
  - 알림 시스템 추가

- [ ] **반응형 디자인 최적화**
  - 모바일 터치 인터페이스 개선
  - 태블릿 레이아웃 최적화
  - 접근성 향상

#### 5. 백엔드 시스템 강화
- [ ] **데이터베이스 스키마 개선**
  - 자세 분석 결과 상세 저장
  - 사용자별 통계 데이터 구조화
  - 성능 최적화

- [ ] **API 엔드포인트 추가** 
  - 자세 히스토리 조회 API
  - 통계 데이터 API
  - 사용자 설정 API

- [ ] **에러 처리 및 로깅**
  - MediaPipe 에러 처리 개선
  - 사용자 친화적 에러 메시지
  - 디버깅 로그 시스템

#### 6. 성능 최적화
- [ ] **실시간 처리 성능 향상**
  - FPS 최적화 (목표: 30fps 이상)
  - 메모리 사용량 최적화
  - CPU 사용률 개선

- [ ] **네트워크 최적화**
  - 이미지 압축 알고리즘 적용
  - WebSocket 연결 안정성 향상
  - 대역폭 사용량 최적화

### 🔧 기술적 고려사항

#### MediaPipe 최적화
- Pose 모델 설정 조정
- 랜드마크 필터링 최적화
- 프레임 스킵 로직 개선

#### 브라우저 호환성
- Chrome, Firefox, Safari 테스트
- 모바일 브라우저 최적화
- WebRTC API 호환성 확인

#### 보안 및 개인정보
- 웹캠 데이터 보안 강화
- 사용자 데이터 암호화
- GDPR 준수 확인

### 📊 성능 목표
- **실시간 분석**: 30fps 이상 유지
- **정확도**: 90% 이상의 자세 감지 정확도
- **응답 시간**: 100ms 이하의 분석 응답
- **안정성**: 99% 이상의 시스템 가동률

### 🚀 다음 단계
1. 팀원 전체 테스트를 통한 다양한 사람 데이터 수집
2. 머신러닝 모델 개발 및 학습
3. 자세 분석 알고리즘 테스트 및 검증
4. 사용자 테스트 진행
5. 피드백 수집 및 반영
6. 최종 배포 준비

---

## 완료된 작업

### 2025-06-29 19:30
- ✅ **머신러닝 모델 개발 및 실시간 분류 시스템 구현**

#### 📊 데이터 기반 모델 설계 의사결정
**데이터셋 분석 결과:**
- 총 395개 레코드 중 필터링된 289개 데이터 활용
- 라벨 분포: 정면(124개), 측면(165개) - 측면 데이터가 약간 많음
- 33개 MediaPipe 랜드마크 좌표를 기반으로 특징 추출

#### 🏗️ 모델 아키텍처 설계 과정

**1. 모델 복잡도별 3가지 버전 개발**

**A. pose_classification_model.py (고정밀 분석용)**
- **특징**: 70개 이상의 특징 (모든 랜드마크 + 다양한 각도/비율/대칭성)
- **모델**: Random Forest + SVM 비교 후 최적 모델 선택
- **용도**: 연구 및 정밀 분석
- **단점**: 실시간 시스템에는 과적합 위험 및 느린 처리 속도

**B. pose_classifier_robust.py (실시간 최적화용) ⭐**
- **특징**: 28개 특징으로 제한 (상체 중심 랜드마크 0-12번만)
- **모델**: Random Forest (과적합 방지 파라미터 적용)
- **성능**: 교차 검증 99.1% (±2.1%), 테스트 정확도 98.3%
- **장점**: 빠른 예측 속도, 안정적인 성능, 과적합 방지

**C. pose_classifier_simple.py (빠른 테스트용)**
- **특징**: 68개 특징, 기본적인 분류만 수행
- **모델**: Random Forest (기본 설정)
- **용도**: 빠른 프로토타이핑 및 테스트

#### 🎯 실시간 시스템 최적화 의사결정

**왜 robust 모델을 선택했는가?**

1. **성능 vs 속도 균형**
   - 정면/측면 분류는 복잡한 특징이 불필요
   - 상체 중심 랜드마크만으로도 충분한 정확도 달성
   - 28개 특징으로 98.3% 정확도 달성

2. **과적합 방지**
   - `max_depth=10`, `min_samples_split=5` 등 제한적 파라미터
   - 교차 검증으로 일반화 성능 검증
   - 실제 환경에서 안정적인 성능 보장

3. **실시간 처리 최적화**
   - 특징 수 제한으로 예측 속도 향상
   - Pickle 형태로 모델 저장/로드하여 효율성 증대
   - 웹캠 스트림 처리에 적합한 경량화

#### 🔧 특징 엔지니어링 전략

**핵심 특징 선택 기준:**
1. **정규화된 좌표 (26개)**: 어깨 중심 기준, 어깨 너비로 스케일링
2. **각도 특징 (1개)**: 목-어깨-팔꿈치 각도
3. **어깨 비율 (1개)**: 어깨 너비/높이 비율
4. **대칭성 특징 (1개)**: 좌우 어깨 대칭성

**특징 중요도 분석 결과:**
- `norm_landmark_3_x` (23.07%): 코의 x좌표가 가장 중요
- `norm_landmark_7_x`, `norm_landmark_11_x`: 귀와 어깨 좌표가 다음으로 중요
- x좌표가 y좌표보다 더 중요한 특징으로 나타남 (정면/측면 구분에 효과적)

#### 🚀 실시간 분류 시스템 구현

**realtime_pose_classifier.py 주요 기능:**
- **한글 폰트 렌더링**: PIL 기반 한글 텍스트 표시
- **실시간 예측**: 30fps 웹캠 스트림에서 실시간 분류
- **시각적 피드백**: 정면(초록), 측면(파랑) 색상 구분
- **신뢰도 표시**: 예측 확률을 퍼센트로 표시
- **좌우반전**: 거울 모드로 자연스러운 사용자 경험
- **디버깅 정보**: 주요 랜드마크 좌표 및 어깨 거리 표시

**성능 최적화 기법:**
- 10프레임마다 예측 수행으로 CPU 부하 감소
- 1초마다 콘솔 출력으로 로그 스팸 방지
- 예외 처리로 안정성 확보
- 여러 카메라 자동 감지 및 연결

#### 📈 모델 성능 평가 결과

**혼동 행렬 분석:**
- 정면 → 정면: 96% 정밀도, 100% 재현율
- 측면 → 측면: 100% 정밀도, 97% 재현율
- 전체 정확도: 98.3%

**교차 검증 결과:**
- 5-fold CV 평균: 99.1% (±2.1%)
- 안정적인 일반화 성능 확인

#### 🔄 시스템 통합 및 테스트

**모델 저장/로드 시스템:**
```python
# 모델 학습 후 Pickle 저장
with open('pose_classifier_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'is_trained': True
    }, f)
```

**실시간 예측 파이프라인:**
1. MediaPipe로 33개 랜드마크 추출
2. 28개 특징으로 변환 (정규화, 각도, 비율, 대칭성)
3. StandardScaler로 정규화
4. Random Forest 모델로 예측
5. 결과 시각화 및 피드백

#### 🎯 향후 개선 방향

**데이터 다양성 확보:**
- 현재 P1 데이터만 사용 → 팀원 전체 데이터 수집 필요
- 체형별, 연령대별 다양한 데이터로 모델 일반화 성능 향상
- 정면에서 고개 숙이는 동작, 고개 드는 동작, 뒤 돌아서는 동작에 대한 데이터셋 확보 후 모델 학습에 추가 필요

**모델 고도화:**
- 앙상블 기법 적용 (Voting, Stacking)
- 딥러닝 모델 (CNN, LSTM) 실험
- 온라인 학습으로 실시간 모델 업데이트

**시스템 안정성:**
- 에러 처리 강화
- 메모리 누수 방지
- 성능 모니터링 시스템 구축

**🔧 관련 기술 요소**
- MediaPipe Pose 33개 랜드마크 추출
- scikit-learn Random Forest 분류기
- OpenCV 웹캠 스트림 처리
- PIL 한글 폰트 렌더링
- Pickle 모델 직렬화/역직렬화
- 실시간 특징 추출 및 정규화

**📌 기술적 성과**
- 98.3% 정확도의 실시간 정면/측면 분류 모델 개발
- 30fps 실시간 처리 성능 달성
- 경량화된 특징 엔지니어링으로 과적합 방지
- 사용자 친화적인 한글 인터페이스 구현

### 2025-06-28 23:00
- ✅ **자세 라벨링 기반 랜드마크 데이터 수집 기능 구현**
  - 정면/측면 상태 감지 정밀도 향상을 위해 MediaPipe의 랜드마크 좌표 데이터를 DB에 저장하는 시스템 개발
  - 10프레임마다 1회 분석 및 저장하여 시스템 리소스 낭비를 줄이고 효율성 향상
  - SQLite 기반 pose_landmarks 테이블 생성: timestamp, label(자세 상태: 1=정면, 2=측면, 3=제외), landmark_0_x ~ landmark_32_y까지 총 66개 좌표 저장
  - 실시간 키보드 입력 기반 라벨링 기능 추가: 1: 정면, 2: 측면, 3: 제외 상태
  - **visibility가 0.5 미만인 관절은 (-1, -1)로 저장하여 학습 데이터 품질 확보**
  - **person_id 컬럼 추가로 여러 사람(P1, P2, P3...) 데이터 구분 가능**
  - **키보드 'P' 입력으로 사람 식별자 변경 기능**
  - **사람별, 라벨별로 분리된 CSV 파일 생성 및 test/data 폴더에 저장**
  - 수집된 데이터는 향후 정면/측면 분류 머신러닝 모델 학습용 데이터셋으로 활용 예정

- ✅ **MediaPipe Pose v1/v2 랜드마크 비교 자료조사**
  - Pose v1: 총 33개 랜드마크 (현재 사용 중)
  - Pose v2: 총 38개 랜드마크 (v1 + 5개 추가)
  - v2 추가 포인트: 골반 중심(MIDHIP) 및 발 앞쪽(FOREFOOT) 관련 포인트
  - v2는 하체 균형 및 보행 정렬 분석에 더 정밀한 분석 가능
  - 향후 자세 분석 정확도 향상을 위해 v2 업그레이드 고려

**🔧 관련 기술 요소**
- MediaPipe Pose 33개 랜드마크 추출
- OpenCV 웹캠 영상 처리 및 키보드 이벤트
- SQLite3로 시계열 자세 데이터 저장
- 10프레임 간격 처리 로직으로 성능 최적화
- 감지 신뢰도(visibility) 기반 결측값 처리 로직 적용

**📌 향후 활용 계획**
- 수집된 데이터를 기반으로 정면/측면 분류 모델(KNN, SVM, 랜덤포레스트 등) 학습
- 정면 vs 측면 실시간 자동 분류 기능 적용
- 수동 라벨링 없이도 자세 상태 판별 가능하도록 시스템 자동화
- 현재 v1 랜드마크를 활용한 더 정교한 자세 분석 알고리즘 개발

### 2025-06-27
- ✅ 기존 Canvas 기반 실시간 스켈레톤 시각화 코드를 `PoseStateManager` 기반의 상태 관리 흐름에 통합
- ✅ 거북목 진단 기준 조사
- ✅ 자세 분석 알고리즘 고도화를 위해 기존 코드를 재검토  
- ✅ 자세 단계를 나눠 실측 테스트 및 결과 캡처 기반 분석을 병행

### 2025-06-26
- ✅ 실시간 스켈레톤 시각화 시스템 구현
- ✅ Canvas 오버레이 시스템 추가
- ✅ 랜드마크 데이터 전송 기능 구현
- ✅ 상태 관리 시스템 완성
- ✅ 반응형 UI/UX 디자인 구현
- ✅ 한글 폰트 렌더링 시스템 구축

---

## 참고 자료
- MediaPipe Pose Documentation
- Flask WebSocket Implementation
- HTML5 Canvas API Reference
- WebRTC API Documentation
- scikit-learn Machine Learning Documentation
- OpenCV Python Tutorials

---

#### 🚀 실제 서비스 환경에서의 모델 개선 전략

**사용자 동의 기반 데이터 수집 시스템 설계**

**1. 데이터 수집 아키텍처**
```python
# 비동기 데이터 수집 파이프라인
class AsyncDataCollector:
    def __init__(self):
        self.data_queue = asyncio.Queue()
        self.processing_task = None
        
    async def collect_user_data(self, landmarks, prediction, user_consent):
        """사용자 동의 시 데이터 수집"""
        if user_consent and self.validate_data_quality(landmarks):
            await self.data_queue.put({
                'timestamp': time.time(),
                'landmarks': landmarks,
                'prediction': prediction,
                'user_id': self.get_anonymous_user_id(),
                'session_id': self.get_session_id()
            })
    
    async def process_collected_data(self):
        """백그라운드에서 수집된 데이터 처리"""
        while True:
            try:
                data = await self.data_queue.get()
                await self.save_to_database(data)
                await self.update_model_if_needed()
            except Exception as e:
                logger.error(f"데이터 처리 오류: {e}")
```

**2. 분기 처리 전략**

**A. 실시간 분석 vs 데이터 수집 분리**
```python
# 메인 분석 스레드 (실시간)
def realtime_analysis_pipeline():
    while True:
        landmarks = extract_landmarks(frame)
        prediction = model.predict(landmarks)
        display_feedback(prediction)
        
        # 비동기로 데이터 수집 (별도 스레드)
        if user_consent:
            asyncio.create_task(
                data_collector.collect_user_data(landmarks, prediction)
            )

# 데이터 수집 스레드 (백그라운드)
async def background_data_processing():
    while True:
        await data_collector.process_collected_data()
        await asyncio.sleep(60)  # 1분마다 배치 처리
```

**B. 사용자 동의 상태별 처리**
```python
class ConsentManager:
    def __init__(self):
        self.user_consents = {}  # user_id -> consent_status
        
    def check_consent(self, user_id):
        return self.user_consents.get(user_id, False)
    
    def update_consent(self, user_id, consent):
        self.user_consents[user_id] = consent
        self.save_consent_to_database(user_id, consent)
    
    def get_anonymous_user_id(self):
        """개인정보 보호를 위한 익명 사용자 ID 생성"""
        return hashlib.sha256(f"{user_session}_{timestamp}".encode()).hexdigest()[:16]
```

**3. 비동기 처리 최적화**

**A. 큐 기반 데이터 처리**
```python
class DataProcessingQueue:
    def __init__(self, max_size=10000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.batch_size = 100
        self.processing_interval = 300  # 5분
        
    async def add_data(self, data):
        try:
            await self.queue.put(data, timeout=1.0)
        except asyncio.QueueFull:
            logger.warning("데이터 큐가 가득참 - 오래된 데이터 삭제")
            await self.queue.get()  # 가장 오래된 데이터 제거
            await self.queue.put(data)
    
    async def process_batch(self):
        """배치 단위로 데이터 처리"""
        batch = []
        while len(batch) < self.batch_size:
            try:
                data = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                batch.append(data)
            except asyncio.TimeoutError:
                break
        
        if batch:
            await self.save_batch_to_database(batch)
            await self.check_model_update_conditions(batch)
```

**B. 메모리 효율적인 데이터 저장**
```python
class EfficientDataStorage:
    def __init__(self):
        self.buffer = []
        self.max_buffer_size = 1000
        
    async def save_data(self, data):
        self.buffer.append(data)
        
        if len(self.buffer) >= self.max_buffer_size:
            await self.flush_buffer()
    
    async def flush_buffer(self):
        """버퍼의 데이터를 데이터베이스에 일괄 저장"""
        if self.buffer:
            await self.batch_insert_to_db(self.buffer)
            self.buffer.clear()
```

**4. 모델 업데이트 전략**

**A. 점진적 모델 업데이트**
```python
class IncrementalModelUpdater:
    def __init__(self, base_model_path):
        self.base_model = self.load_model(base_model_path)
        self.new_data_threshold = 1000  # 1000개 새 데이터마다 업데이트
        self.performance_threshold = 0.95  # 성능 임계값
        
    async def check_update_conditions(self, new_data_count):
        if new_data_count >= self.new_data_threshold:
            await self.update_model_incrementally()
    
    async def update_model_incrementally(self):
        """새 데이터로 모델 점진적 업데이트"""
        new_features, new_labels = await self.prepare_new_training_data()
        
        # 기존 모델에 새 데이터 추가 학습
        self.base_model.partial_fit(new_features, new_labels)
        
        # 성능 검증
        validation_score = self.validate_model_performance()
        if validation_score >= self.performance_threshold:
            await self.save_updated_model()
            logger.info("모델 업데이트 완료")
        else:
            logger.warning("모델 성능 저하 - 업데이트 롤백")
```

**B. A/B 테스트 기반 모델 배포**
```python
class ModelABTesting:
    def __init__(self):
        self.current_model = self.load_production_model()
        self.candidate_model = None
        self.test_ratio = 0.1  # 10% 사용자에게 새 모델 테스트
        
    async def deploy_candidate_model(self, new_model):
        """새 모델을 일부 사용자에게 배포하여 테스트"""
        self.candidate_model = new_model
        
        # 사용자 그룹 분할
        test_users = self.select_test_users(self.test_ratio)
        
        for user_id in test_users:
            await self.assign_model_to_user(user_id, 'candidate')
    
    async def evaluate_model_performance(self):
        """A/B 테스트 결과 평가"""
        current_performance = await self.get_model_performance('current')
        candidate_performance = await self.get_model_performance('candidate')
        
        if candidate_performance > current_performance * 1.05:  # 5% 이상 개선
            await self.promote_candidate_model()
```

**5. 개인정보 보호 및 보안**

**A. 데이터 익명화**
```python
class DataAnonymizer:
    def anonymize_landmarks(self, landmarks):
        """랜드마크 데이터 익명화"""
        # 개인 식별 가능한 정보 제거
        anonymized = []
        for i, (x, y) in enumerate(landmarks):
            # 얼굴 부위 랜드마크는 정규화만 수행
            if i < 10:  # 얼굴 랜드마크
                norm_x = (x - 0.5) * 2  # -1 ~ 1 범위로 정규화
                norm_y = (y - 0.5) * 2
                anonymized.extend([norm_x, norm_y])
            else:
                anonymized.extend([x, y])
        return anonymized
    
    def generate_session_id(self):
        """세션별 고유 ID 생성"""
        return str(uuid.uuid4())
```

**B. 동의 관리 시스템**
```python
class ConsentManagement:
    def __init__(self):
        self.consent_db = {}
        
    async def request_consent(self, user_id):
        """사용자에게 데이터 수집 동의 요청"""
        consent_ui = self.create_consent_ui()
        user_choice = await self.show_consent_dialog(consent_ui)
        
        if user_choice == 'accept':
            await self.save_consent(user_id, True, time.time())
            return True
        else:
            await self.save_consent(user_id, False, time.time())
            return False
    
    async def revoke_consent(self, user_id):
        """동의 철회 처리"""
        await self.save_consent(user_id, False, time.time())
        await self.delete_user_data(user_id)
```

**6. 성능 모니터링 및 알림**

**A. 실시간 성능 모니터링**
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'prediction_accuracy': [],
            'processing_time': [],
            'data_collection_rate': [],
            'model_update_frequency': []
        }
        
    async def monitor_system_performance(self):
        """시스템 성능 실시간 모니터링"""
        while True:
            current_accuracy = await self.calculate_current_accuracy()
            processing_time = await self.measure_processing_time()
            
            self.metrics['prediction_accuracy'].append(current_accuracy)
            self.metrics['processing_time'].append(processing_time)
            
            # 성능 임계값 체크
            if current_accuracy < 0.9:  # 90% 미만
                await self.send_alert('모델 성능 저하 감지')
            
            if processing_time > 100:  # 100ms 초과
                await self.send_alert('처리 시간 지연 감지')
            
            await asyncio.sleep(60)  # 1분마다 체크
```

**B. 자동 복구 시스템**
```python
class AutoRecoverySystem:
    async def handle_model_degradation(self):
        """모델 성능 저하 시 자동 복구"""
        if await self.detect_performance_degradation():
            # 이전 버전 모델로 롤백
            await self.rollback_to_previous_model()
            
            # 새 데이터로 모델 재학습
            await self.retrain_model_with_recent_data()
            
            # 성능 검증 후 재배포
            if await self.validate_model_performance():
                await self.deploy_updated_model()
```

**🔧 기술적 구현 고려사항**

**1. 비동기 처리 최적화**
- `asyncio` 기반 비동기 데이터 처리
- 큐 기반 배치 처리로 시스템 부하 분산
- 메모리 효율적인 데이터 버퍼링

**2. 확장성 고려**
- 마이크로서비스 아키텍처 적용 가능성
- 데이터베이스 샤딩 및 파티셔닝
- 로드 밸런싱을 통한 서버 분산

**3. 보안 및 개인정보 보호**
- GDPR, CCPA 등 개인정보보호법 준수
- 데이터 암호화 및 익명화
- 동의 관리 및 철회 시스템

**4. 모니터링 및 운영**
- 실시간 성능 모니터링
- 자동 복구 및 롤백 시스템
- A/B 테스트 기반 안전한 모델 배포

**📊 예상 효과**
- **데이터 다양성**: 실제 사용자 데이터로 모델 일반화 성능 향상
- **지속적 개선**: 온라인 학습으로 모델 성능 지속 향상
- **사용자 경험**: 개인화된 자세 분석 서비스 제공
- **서비스 안정성**: 자동화된 모니터링 및 복구 시스템

**🎯 다음 단계**
1. 사용자 동의 관리 시스템 구현
2. 비동기 데이터 수집 파이프라인 구축
3. 점진적 모델 업데이트 시스템 개발
4. A/B 테스트 프레임워크 구축
5. 성능 모니터링 및 알림 시스템 구현