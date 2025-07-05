#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단순한 동영상 촬영 스크립트

실행과 동시에 촬영이 시작되고, 'q' 키를 누르면 저장 후 종료됩니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.modules.video_recorder import VideoRecorder


def main():
    """메인 함수"""
    print("🎥 단순한 동영상 촬영을 시작합니다...")
    print("📝 'q' 키를 누르면 촬영이 종료되고 파일이 저장됩니다.")
    print()
    
    # VideoRecorder 인스턴스 생성 및 촬영 시작
    recorder = VideoRecorder()
    recorder.start_recording()
    
    print("✅ 촬영이 완료되었습니다!")


if __name__ == "__main__":
    main() 