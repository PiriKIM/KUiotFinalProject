#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동영상 재생 실행 스크립트

동영상 파일을 재생합니다. 'q' 키를 누르면 재생이 종료됩니다.
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.modules.video_player import VideoPlayer


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='동영상 재생')
    parser.add_argument('video_path', help='재생할 동영상 파일 경로')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.video_path).exists():
        print(f"❌ 동영상 파일을 찾을 수 없습니다: {args.video_path}")
        return 1
    
    print("🎬 동영상 재생을 시작합니다...")
    print(f"📁 파일: {args.video_path}")
    print("📝 'q' 키를 누르면 재생이 종료됩니다.")
    print()
    
    # VideoPlayer 인스턴스 생성 및 재생 시작
    player = VideoPlayer()
    player.play_video(args.video_path)
    
    print("✅ 재생이 완료되었습니다!")
    return 0


if __name__ == "__main__":
    exit(main()) 