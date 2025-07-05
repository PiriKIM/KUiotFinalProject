#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자세 분석 파이프라인 설정 파일

이 파일은 프로젝트 전체에서 사용되는 설정값들을 정의합니다.
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# 동영상 촬영 설정
# =============================================================================
VIDEO_SETTINGS = {
    'duration': 20,              # 촬영 시간 (초)
    'fps': 30,                   # 프레임 레이트
    'resolution': (1280, 720),   # 해상도 (width, height)
    'camera_index': 0,           # 웹캠 인덱스
    'output_filename': 'video.mp4',  # 출력 파일명
    'codec': 'mp4v',             # 비디오 코덱
    'mirror': True,              # 미러링 여부 (사용자 피드백용)
}

# =============================================================================
# 프레임 추출 설정
# =============================================================================
FRAME_SETTINGS = {
    'frame_count': 50,           # 추출할 프레임 수
    'output_format': 'jpg',      # 출력 이미지 형식
    'quality': 95,               # 이미지 품질 (1-100)
}

# =============================================================================
# MediaPipe 설정
# =============================================================================
MEDIAPIPE_SETTINGS = {
    'static_image_mode': True,   # 정적 이미지 모드
    'model_complexity': 2,       # 모델 복잡도 (0, 1, 2)
    'enable_segmentation': False, # 세그멘테이션 활성화
    'min_detection_confidence': 0.5,  # 최소 감지 신뢰도
    'min_tracking_confidence': 0.5,   # 최소 추적 신뢰도
}

# =============================================================================
# 각도 계산 설정
# =============================================================================
ANGLE_SETTINGS = {
    'neck_angle_threshold': 15,  # 목각도 임계값 (도)
    'spine_angle_threshold': 10, # 척추각도 임계값 (도)
    'angle_calculation_method': 'vector',  # 각도 계산 방법
}

# =============================================================================
# 등급 분류 설정
# =============================================================================
GRADE_SETTINGS = {
    'grade_count': 10,           # 등급 수
    'primary_angle': 'neck',     # 주요 평가 각도 ('neck' 또는 'spine')
    'grade_labels': {            # 등급 라벨
        1: 'A', 2: 'B', 3: 'B',
        4: 'B', 5: 'C', 6: 'C',
        7: 'C', 8: 'C', 9: 'C',
        10: 'C'
    }
}

# =============================================================================
# 파일 경로 설정
# =============================================================================
PATHS = {
    'data_dir': PROJECT_ROOT / 'data',
    'raw_video_dir': PROJECT_ROOT / 'data' / 'raw',
    'frames_dir': PROJECT_ROOT / 'data' / 'frames',
    'landmarks_dir': PROJECT_ROOT / 'data' / 'landmarks',
    'angles_dir': PROJECT_ROOT / 'data' / 'angles',
    'grades_dir': PROJECT_ROOT / 'data' / 'grades',
    'results_dir': PROJECT_ROOT / 'data' / 'results',
    'visualization_dir': PROJECT_ROOT / 'data' / 'results' / 'visualization',
    'stage_images_dir': PROJECT_ROOT / 'data' / 'grades' / 'stage_images',
}

# =============================================================================
# 로깅 설정
# =============================================================================
LOGGING_SETTINGS = {
    'level': 'INFO',             # 로그 레벨
    'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
    'file': PROJECT_ROOT / 'logs' / 'pipeline.log',
    'rotation': '1 day',         # 로그 파일 로테이션
    'retention': '7 days',       # 로그 보관 기간
}

# =============================================================================
# 시각화 설정
# =============================================================================
VISUALIZATION_SETTINGS = {
    'figure_size': (12, 8),      # 그래프 크기
    'dpi': 100,                  # 해상도
    'style': 'seaborn-v0_8',     # matplotlib 스타일
    'color_palette': 'viridis',  # 색상 팔레트
    'save_format': 'png',        # 저장 형식
}

# =============================================================================
# 테스트 설정
# =============================================================================
TEST_SETTINGS = {
    'test_data_dir': PROJECT_ROOT / 'tests' / 'test_data',
    'mock_video_duration': 5,    # 테스트용 동영상 길이
    'mock_frame_count': 10,      # 테스트용 프레임 수
}

# =============================================================================
# 유틸리티 함수
# =============================================================================
def ensure_directories():
    """필요한 디렉토리들을 생성합니다."""
    for path in PATHS.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    
    # 로그 디렉토리 생성
    log_dir = PROJECT_ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)

def get_timestamp():
    """현재 타임스탬프를 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_output_filename(prefix: str, extension: str = 'mp4') -> str:
    """타임스탬프가 포함된 출력 파일명을 생성합니다."""
    timestamp = get_timestamp()
    return f"{prefix}_{timestamp}.{extension}"

# =============================================================================
# 초기화
# =============================================================================
# 프로젝트 시작 시 필요한 디렉토리들을 생성
ensure_directories() 