# 좋은 자세 감지 시스템 (Good Posture Detector)

MediaPipe를 활용한 실시간 좋은 자세 감지 및 분석 시스템입니다.

## 📁 디렉토리 구조

```
good_posture_detector/
├── __init__.py                    # 패키지 초기화
├── config.py                      # 시스템 설정 파일
├── reference_analyzer.py          # 기준 이미지 분석 (좋은/나쁜 자세)
├── pattern_learner.py             # 좋은 자세 패턴 학습
├── feature_extractor.py           # 자세 특징 추출
├── realtime_detector.py           # 실시간 자세 감지
├── visualization.py               # 결과 시각화
├── data/                          # 데이터 저장소
│   ├── reference_landmarks/       # 추출된 기준 랜드마크
│   ├── good_posture_samples/      # 좋은 자세 샘플
│   └── bad_posture_samples/       # 나쁜 자세 샘플
├── models/                        # 학습된 모델 저장
│   └── __init__.py
├── tests/                         # 테스트 파일들
│   ├── __init__.py
│   ├── test_reference_analyzer.py
│   ├── test_pattern_learner.py
│   └── test_realtime_detector.py
└── utils/                         # 유틸리티 함수들
    ├── __init__.py
    ├── alignment_calculator.py    # 정렬도 계산
    ├── symmetry_calculator.py     # 대칭성 계산
    └── angle_calculator.py        # 각도 계산
```

## 🔧 주요 컴포넌트

### 1. 기준 이미지 분석 (`reference_analyzer.py`)
- 좋은 자세와 나쁜 자세 이미지에서 MediaPipe 랜드마크 추출
- 기준 샘플 데이터베이스 구축
- 이미지 품질 검증 및 전처리

### 2. 패턴 학습 (`pattern_learner.py`)
- 좋은 자세의 통계적 패턴 학습
- 목-어깨-골반 정렬 패턴 분석
- 대칭성 및 각도 패턴 학습
- 임계값 자동 설정

### 3. 특징 추출 (`feature_extractor.py`)
- 목 정렬도 계산
- 척추 직선성 측정
- 어깨 대칭성 분석
- 골반 정렬도 평가
- 전체 균형 점수 계산

### 4. 실시간 감지 (`realtime_detector.py`)
- 웹캠을 통한 실시간 자세 감지
- 좋은 자세 여부 실시간 판정
- 점수 기반 평가 시스템
- 피드백 생성

### 5. 시각화 (`visualization.py`)
- 랜드마크 및 연결선 표시
- 자세 점수 시각화
- 피드백 메시지 표시
- 결과 저장 및 로깅

## 🛠️ 유틸리티 함수들

### `alignment_calculator.py`
- 수직 정렬도 계산
- 수평 정렬도 계산
- 전체 정렬 점수 계산

### `symmetry_calculator.py`
- 어깨 대칭성 계산
- 골반 대칭성 계산
- 좌우 균형 점수 계산

### `angle_calculator.py`
- 목 각도 계산
- 척추 각도 계산
- 어깨 기울기 계산
- 골반 기울기 계산

## 📊 데이터 구조

### 기준 랜드마크 데이터
```json
{
  "image_path": "path/to/image.png",
  "posture_type": "good",  // "good" or "bad"
  "landmarks": [
    {"x": 0.5, "y": 0.3, "visibility": 0.9},
    // ... 33개 랜드마크
  ],
  "features": {
    "neck_alignment": 0.95,
    "spine_straightness": 0.88,
    "shoulder_symmetry": 0.92,
    "pelvic_alignment": 0.90
  }
}
```

### 학습된 패턴 모델
```json
{
  "good_posture_thresholds": {
    "neck_alignment": 0.85,
    "spine_straightness": 0.80,
    "shoulder_symmetry": 0.85,
    "pelvic_alignment": 0.80,
    "overall_score": 0.82
  },
  "feature_weights": {
    "neck": 0.3,
    "spine": 0.25,
    "shoulder": 0.25,
    "pelvic": 0.2
  }
}
```

## 🚀 사용 방법

### 1. 기준 이미지 분석
```python
from good_posture_detector.reference_analyzer import ReferenceAnalyzer

analyzer = ReferenceAnalyzer()
analyzer.analyze_reference_images("data/image/good_posture/")
analyzer.analyze_reference_images("data/image/bad_posture/")
```

### 2. 패턴 학습
```python
from good_posture_detector.pattern_learner import PatternLearner

learner = PatternLearner()
learner.learn_good_posture_patterns()
learner.save_patterns("models/good_posture_patterns.json")
```

### 3. 실시간 감지
```python
from good_posture_detector.realtime_detector import RealTimeDetector

detector = RealTimeDetector()
detector.start_detection()
```

## 📈 평가 지표

- **정확도 (Accuracy)**: 좋은 자세 감지 정확도
- **민감도 (Sensitivity)**: 좋은 자세를 좋은 자세로 올바르게 분류
- **특이도 (Specificity)**: 나쁜 자세를 나쁜 자세로 올바르게 분류
- **F1-Score**: 정밀도와 재현율의 조화평균

## 🔄 개발 워크플로우

1. **데이터 수집**: 좋은/나쁜 자세 이미지 수집
2. **기준 분석**: MediaPipe로 랜드마크 추출
3. **패턴 학습**: 좋은 자세의 특징 패턴 학습
4. **모델 검증**: 테스트 데이터로 성능 평가
5. **실시간 구현**: 웹캠 기반 실시간 감지
6. **최적화**: 성능 개선 및 피드백 시스템

## 📝 의존성

- `mediapipe`: 포즈 랜드마크 추출
- `opencv-python`: 이미지 처리 및 웹캠
- `numpy`: 수치 계산
- `matplotlib`: 시각화
- `scikit-learn`: 머신러닝 (선택사항)

## 🤝 기존 시스템과의 통합

이 시스템은 `test_grade` 프로젝트의 기존 자세 등급 시스템과 통합하여 사용할 수 있습니다:

- 기존 `pose_grade_data.db` 활용
- `PostureAnalyzer` 클래스와 연동
- 자동 등급 판별 시스템 강화 