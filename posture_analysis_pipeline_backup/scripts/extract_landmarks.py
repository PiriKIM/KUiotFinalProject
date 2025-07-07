#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
랜드마크 추출 실행 스크립트

프레임 이미지에서 MediaPipe를 사용하여 랜드마크를 추출하고,
카메라 위치를 자동 감지하여 측면별 랜드마크를 선택합니다.

# 실행 방법 (P2로 실행)
# python3 scripts/extract_landmarks.py /home/woo/KUiotFinalProject/posture_analysis_pipeline/frames_posture_video_20250706_173222 --subject P2

"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.modules.landmark_extractor import LandmarkExtractor


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='프레임에서 랜드마크 추출')
    parser.add_argument('frames_dir', help='프레임 이미지 디렉토리 경로')
    parser.add_argument('--output', '-o', help='출력 디렉토리 (기본값: data/landmarks)')
    parser.add_argument('--subject', '-s', default='P1', help='피사체 ID (기본값: P1)')
    
    args = parser.parse_args()
    
    # 디렉토리 존재 확인
    if not Path(args.frames_dir).exists():
        print(f"❌ 프레임 디렉토리를 찾을 수 없습니다: {args.frames_dir}")
        return 1
    
    print("🎯 랜드마크 추출을 시작합니다...")
    print(f"📁 프레임 디렉토리: {args.frames_dir}")
    print(f"📂 출력 디렉토리: {args.output or 'data/landmarks'}")
    print(f"👤 피사체 ID: {args.subject}")
    print()
    
    # LandmarkExtractor 인스턴스 생성
    extractor = LandmarkExtractor()
    
    # 랜드마크 추출 실행
    extracted_data = extractor.extract_landmarks_from_directory(
        frames_dir=args.frames_dir,
        output_dir=args.output,
        subject_id=args.subject
    )
    
    if extracted_data:
        print(f"✅ 랜드마크 추출 완료!")
        print(f"📊 처리된 프레임: {len(extracted_data)}개")
        
        # 카메라 위치 통계
        camera_positions = {}
        for data in extracted_data:
            pos = data['camera_position']
            camera_positions[pos] = camera_positions.get(pos, 0) + 1
        
        print(f"📷 카메라 위치 통계:")
        for pos, count in camera_positions.items():
            print(f"  - {pos}: {count}개 프레임")
        
        # 출력 파일 경로
        output_path = Path(args.output) if args.output else Path("data/landmarks")
        print(f"📄 저장된 파일:")
        print(f"  - CSV: {output_path / 'raw_landmarks.csv'}")
        print(f"  - JSON: {output_path / 'raw_landmarks.json'}")
        
    else:
        print("❌ 랜드마크 추출 실패")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 