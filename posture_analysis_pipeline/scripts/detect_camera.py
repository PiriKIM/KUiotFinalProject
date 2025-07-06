#!/usr/bin/env python3
"""
카메라 방향 감지 실행 스크립트

랜드마크 CSV 파일을 읽어서 카메라 방향을 감지하고 결과를 출력합니다.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modules.camera_detector import detect_camera_position_from_csv

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='카메라 방향 감지')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/landmarks/raw_landmarks.csv',
        help='입력 랜드마크 CSV 파일 경로'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/landmarks/landmarks_with_camera.csv',
        help='출력 CSV 파일 경로 (선택사항)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='결과를 파일로 저장하지 않음'
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 입력 파일 경로 확인
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return 1
    
    # 출력 파일 경로 설정
    output_path = None
    if not args.no_save:
        output_path = Path(args.output)
        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 카메라 방향 감지
        logger.info(f"카메라 방향 감지를 시작합니다: {input_path}")
        camera_position = detect_camera_position_from_csv(
            str(input_path), 
            str(output_path) if output_path else None
        )
        
        print(f"\n=== 카메라 방향 감지 결과 ===")
        print(f"입력 파일: {input_path}")
        print(f"감지된 방향: {camera_position}")
        
        if output_path:
            print(f"결과 저장: {output_path}")
        
        print(f"\n방향 설명:")
        if camera_position == 'front':
            print("- 정면 촬영: 카메라가 사람의 정면에서 촬영")
        elif camera_position == 'left_side':
            print("- 왼쪽 측면 촬영: 카메라가 사람의 왼쪽 측면에서 촬영")
        elif camera_position == 'right_side':
            print("- 오른쪽 측면 촬영: 카메라가 사람의 오른쪽 측면에서 촬영")
        elif camera_position == 'back':
            print("- 후면 촬영: 카메라가 사람의 뒤에서 촬영")
        else:
            print("- 알 수 없는 방향: 감지에 실패했습니다")
        
        return 0
        
    except Exception as e:
        logger.error(f"카메라 방향 감지 중 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 