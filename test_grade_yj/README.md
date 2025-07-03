# 자세 등급 판별 모델 프로젝트

## 프로젝트 개요
MediaPipe를 활용한 실시간 자세 분석 및 등급 판별 시스템

## 디렉토리 구조
```
test_grade/
├── collector/          # 데이터 수집 관련 코드
├── ml_models/         # 머신러닝 모델 관련 코드
│   ├── models/        # 훈련된 모델 파일들
│   ├── results/       # 모델 성능 결과 (그래프, 메트릭)
│   └── configs/       # 모델 설정 파일들
├── data/              # 데이터셋
│   ├── raw/           # 원본 데이터
│   ├── processed/     # 전처리된 데이터
│   └── augmented/     # 증강된 데이터
├── analysis/          # 데이터 분석 및 시각화
├── utils/             # 유틸리티 함수들
└── README.md
```

## 주요 기능
1. **데이터 수집**: 웹캠을 통한 실시간 자세 데이터 수집
2. **자세 분석**: 목, 어깨, 척추, 골반 등 각 부위별 자세 분석
3. **등급 분류**: A, B, C, D 등급으로 자세 분류
4. **실시간 모니터링**: 실시간 자세 피드백 제공

## 사용법
1. 데이터 수집: `python collector/pose_grade_collector.py`
2. 모델 훈련: `python ml_models/train_grade_model.py`
3. 실시간 테스트: `python analysis/realtime_grade_test.py` 