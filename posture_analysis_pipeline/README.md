# 🎯 자세 분석 파이프라인 (Posture Analysis Pipeline)

자세 등급 분류를 위한 CVA(Cervical Vertebral Angle) 기준 각도를 정량적으로 정의하고, **의학적 철학 기반 3단계 등급 시스템(A/B/C)**을 적용하는 모듈화된 파이프라인입니다.

## 📋 프로젝트 개요

이 프로젝트는 웹캠을 통해 사용자의 자세 변화를 촬영하고, MediaPipe를 활용하여 랜드마크를 추출한 후, CVA 각도를 계산하여 객관적인 자세 등급을 분류하는 7단계 파이프라인을 제공합니다.

### 🎯 주요 목표
- **정량적 자세 평가**: CVA 각도를 통한 객관적 측정
- **의학적 등급 분류**: 누적 데미지 관점의 3단계 등급 시스템
- **측면 자동 감지**: 카메라 위치에 따른 랜드마크 선택
- **모듈화된 구조**: 재사용 가능한 컴포넌트 설계

## 🏗️ 프로젝트 구조

```
posture_analysis_pipeline/
├── README.md                 # 프로젝트 문서
├── requirements.txt          # 의존성 패키지
├── config/                   # 설정 파일
│   ├── settings.py          # 전역 설정값
│   └── paths.py             # 경로 설정
├── src/                      # 소스 코드
│   ├── modules/             # 핵심 기능 모듈
│   │   ├── video_recorder.py        # 1단계: 동영상 촬영
│   │   ├── frame_extractor.py       # 2단계: 프레임 분할
│   │   ├── landmark_extractor.py    # 3단계: 랜드마크 추출
│   │   ├── coordinate_filter.py     # 4단계: 좌표 필터링
│   │   ├── angle_calculator.py      # 5단계: CVA 각도 계산
│   │   ├── grade_classifier.py      # 6단계: 등급 분류
│   │   └── image_selector.py        # 7단계: 이미지 선택
│   ├── utils/               # 공통 유틸리티
│   │   ├── camera_detector.py       # 카메라 위치 감지
│   │   ├── file_manager.py          # 파일 관리
│   │   ├── visualization.py         # 시각화 도구
│   │   └── data_validator.py        # 데이터 검증
│   └── pipeline/            # 파이프라인 관리
│       └── main_pipeline.py         # 전체 워크플로우
├── data/                    # 데이터 저장소
│   ├── raw/                 # 원본 동영상
│   ├── frames/              # 추출된 프레임
│   ├── landmarks/           # 랜드마크 데이터
│   ├── angles/              # 각도 계산 결과
│   ├── grades/              # 등급 분류 결과
│   └── results/             # 최종 결과
├── tests/                   # 테스트 코드
├── scripts/                 # 실행 스크립트
└── docs/                    # 문서
```

## 🔄 7단계 파이프라인

### 1️⃣ **동영상 촬영** (`record_video.py`)
- 웹캠을 통한 자세 변화 촬영 (15-20초)
- 1단계(바른 자세) → 10단계(무너진 자세) 점진적 변화
- `posture_video_2025xxxx_xxxxxx.mp4` 저장

### 2️⃣ **프레임 분할** (`extract_frames.py`)
- 동영상을 등간격으로 50개 프레임 추출
- `frames/frame_01.jpg ~ frame_50.jpg` 저장

### 3️⃣ **랜드마크 추출** (`extract_landmarks.py`) ✅
- MediaPipe Pose로 랜드마크 추출
- 카메라 위치 자동 감지 (왼쪽/오른쪽 측면)
- 측면별 랜드마크 선택 (LEFT_XXX / RIGHT_XXX)
- CSV/JSON 저장: 피사체ID, 프레임명, 좌표값, 카메라 위치

### 4️⃣ **좌표 필터링** (`coordinate_filter.py`)
- 카메라 위치에 따른 측면별 좌표 선택
- CVA 계산용 핵심 좌표만 추출 (귀, 어깨, 골반)
- 필터링된 CSV 생성

### 5️⃣ **CVA 각도 계산** (`angle_calculator.py`)
- **CVA 1**: 귀-어깨 각도 (목각도)
- **CVA 2**: 어깨-골반 각도 (척추각도)
- 벡터 간 각도 계산 방식
- 최종 CSV: 피사체ID, 프레임명, CVA 1, CVA 2

### 6️⃣ **등급 분류** (`grade_classifier.py`)
- CVA 1 기준으로 10단계 등급 구간 설정
- **의학적 철학 기반 3단계 등급 시스템**:
  - **A 등급 (1단계만)**: "이상적인 정렬" - 진정한 '좋은 자세'만 인정
  - **B 등급 (2-4단계)**: "경고 구간" - 짧고 경고성 있는 구간, 즉시 개선 필요
  - **C 등급 (5-10단계)**: "누적 외상 구간" - 경증에서 심각까지 포함하는 누적 외상
- **누적 데미지 관점**: 조금이라도 나쁜 각도는 데미지가 누적되므로 엄격한 기준 적용

### 7️⃣ **이미지 선택** (`image_selector.py`)
- 각 단계별 조건에 해당하는 프레임 이미지 식별
- 단계별 대표 이미지 추출 및 저장
- 결과 시각화

## 🚀 시작하기

### 필수 요구사항
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib

### 설치
```bash
# 저장소 클론
git clone <repository-url>
cd posture_analysis_pipeline

# 의존성 설치
pip install -r requirements.txt
```

### 사용법

#### 전체 파이프라인 실행
```bash
python scripts/run_full_pipeline.py
```

#### 단계별 실행
```bash
# 1단계: 동영상 촬영
python3 scripts/record_video.py

# 2단계: 프레임 분할
python3 scripts/extract_frames.py data/raw/posture_video_20250706_172322.mp4

# 3단계: 랜드마크 추출
python3 scripts/extract_landmarks.py data/frames/
```

#### 결과 시각화
```bash
python scripts/visualize_results.py
```

## 📊 출력 결과

### 데이터 파일
- `data/landmarks/raw_landmarks.csv`: 원본 랜드마크 데이터
- `data/landmarks/filtered_landmarks.csv`: 필터링된 좌표
- `data/angles/cva_angles.csv`: CVA 각도 계산 결과
- `data/grades/grade_classification.csv`: 등급 분류 결과
- `data/results/final_report.csv`: 최종 분석 리포트

### 시각화 결과
- `data/results/visualization/`: 각도 변화 그래프, 등급별 이미지 등

## 📊 등급 분류 시스템

### 🎯 의학적 철학 기반 3단계 등급

이 프로젝트는 **누적 데미지 관점**에서 자세를 평가하는 의학적 철학을 기반으로 합니다:

#### **A 등급 (1단계만) - "이상적인 정렬"**
- **의미**: 진정한 '좋은 자세'만을 인정
- **철학**: 완벽한 정렬 상태만이 건강한 자세로 간주
- **실용적 효과**: 사용자에게 높은 기준 제시

#### **B 등급 (2-4단계) - "경고 구간"**
- **의미**: 짧고 경고성 있는 구간
- **철학**: 즉시 개선이 필요한 상태
- **실용적 효과**: 사용자에게 행동 변화 유도

#### **C 등급 (5-10단계) - "누적 외상 구간"**
- **의미**: 가장 두껍고 포괄적인 구간
- **철학**: 경증에서 심각까지 포함하는 누적 외상
- **실용적 효과**: 조금이라도 나쁜 각도는 데미지 누적

### 💡 설계 철학의 장점
1. **의학적 정확성**: 실제 자세 문제의 누적 효과를 반영
2. **행동 유도**: 명확한 기준으로 사용자 동기부여
3. **실용성**: 복잡한 등급보다 단순하고 명확한 분류
4. **예방적 접근**: 조기 경고 시스템으로 작동

## 🔧 설정

```python
# 동영상 설정
VIDEO_DURATION = 20  # 촬영 시간 (초)
FRAME_COUNT = 50     # 추출할 프레임 수

# 각도 계산 설정
NECK_ANGLE_THRESHOLD = 15  # 목각도 임계값
SPINE_ANGLE_THRESHOLD = 10 # 척추각도 임계값

# 등급 분류 설정 (의학적 철학 기반)
GRADE_COUNT = 10     # 등급 수
PRIMARY_ANGLE = 'neck'  # 주요 평가 각도
GRADE_LABELS = {     # 3단계 등급 라벨
    1: 'A', 2: 'B', 3: 'B', 4: 'B', 5: 'C',
    6: 'C', 7: 'C', 8: 'C', 9: 'C', 10: 'C'
}
```

## 🧪 테스트

```bash
# 전체 테스트 실행
python -m pytest tests/

# 특정 모듈 테스트
python -m pytest tests/test_video_recorder.py
```

## 📝 API 문서

각 모듈의 상세한 API 문서는 `docs/api_documentation.md`를 참조하세요.

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

프로젝트에 대한 문의사항이나 버그 리포트는 Issues 탭을 이용해 주세요.

---

**개발자**: [Umi]  
**버전**: 1.0.0  
**최종 업데이트**: 2024년 7월 