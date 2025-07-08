#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프레임 추출 실행 스크립트

동영상을 50개 프레임으로 등간격 분할하여 저장합니다.
시작 시간과 끝 시간을 지정하여 원하는 구간만 추출할 수 있습니다.
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.modules.frame_extractor import FrameExtractor


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='동영상 프레임 추출')
    parser.add_argument('video_path', help='동영상 파일 경로')
    parser.add_argument('--output', '-o', help='출력 디렉토리 (기본값: 자동 생성)')
    parser.add_argument('--frames', '-f', type=int, default=50, help='추출할 프레임 수 (기본값: 50)')
    parser.add_argument('--start-time', '-s', type=float, help='시작 시간 (초, 기본값: 0초)')
    parser.add_argument('--end-time', '-e', type=float, help='끝 시간 (초, 기본값: 동영상 끝)')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.video_path).exists():
        print(f"❌ 동영상 파일을 찾을 수 없습니다: {args.video_path}")
        return 1
    
    print("🖼️ 프레임 추출을 시작합니다...")
    print(f"📁 동영상: {args.video_path}")
    print(f"📂 출력: {args.output or '자동 생성'}")
    print(f"🎯 프레임 수: {args.frames}")
    
    # 시간 구간 정보 출력
    if args.start_time is not None or args.end_time is not None:
        start_time = args.start_time if args.start_time is not None else 0
        end_time = args.end_time if args.end_time is not None else "동영상 끝"
        print(f"⏰ 시간 구간: {start_time}초 ~ {end_time}초")
    else:
        print(f"⏰ 시간 구간: 전체 동영상")
    
    print()
    
    # FrameExtractor 인스턴스 생성
    extractor = FrameExtractor()
    
    # 동영상 정보 출력
    total_frames, fps, duration = extractor.get_video_info(args.video_path)
    print(f"📊 동영상 정보:")
    print(f"  - 총 프레임: {total_frames}")
    print(f"  - FPS: {fps:.1f}")
    print(f"  - 재생 시간: {duration:.1f}초")
    print()
    
    # 시간 구간 검증
    if args.start_time is not None and args.start_time < 0:
        print(f"❌ 시작 시간은 0 이상이어야 합니다: {args.start_time}")
        return 1
    
    if args.end_time is not None and args.end_time > duration:
        print(f"❌ 끝 시간은 동영상 길이({duration:.1f}초) 이하여야 합니다: {args.end_time}")
        return 1
    
    if args.start_time is not None and args.end_time is not None:
        if args.start_time >= args.end_time:
            print(f"❌ 시작 시간({args.start_time})이 끝 시간({args.end_time})보다 작아야 합니다.")
            return 1
    
    # 프레임 추출 (시간 구간 지정)
    saved_frames = extractor.extract_frames_with_time_range(
        args.video_path, 
        args.output, 
        args.frames,
        start_time=args.start_time,
        end_time=args.end_time
    )
    
    if saved_frames:
        print(f"✅ 프레임 추출 완료!")
        print(f"📁 저장된 프레임: {len(saved_frames)}개")
        print(f"📂 저장 위치: {Path(saved_frames[0]).parent}")
        
        # 처음 5개와 마지막 5개 프레임 경로 출력
        if len(saved_frames) > 10:
            print(f"📋 저장된 파일들:")
            for i, frame_path in enumerate(saved_frames[:5]):
                print(f"  - {Path(frame_path).name}")
            print(f"  ... ({len(saved_frames)-10}개 생략) ...")
            for frame_path in saved_frames[-5:]:
                print(f"  - {Path(frame_path).name}")
        else:
            print(f"📋 저장된 파일들:")
            for frame_path in saved_frames:
                print(f"  - {Path(frame_path).name}")
    else:
        print("❌ 프레임 추출 실패")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 