#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단순한 동영상 재생 모듈

이 모듈은 동영상 파일을 재생하는 기능을 제공합니다.
'q' 키를 누르면 재생이 종료됩니다.
"""

import cv2
import time
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class VideoPlayer:
    """
    단순한 동영상 재생 클래스
    
    동영상 파일을 재생하고 제어할 수 있습니다.
    """
    
    def __init__(self):
        """VideoPlayer 초기화"""
        self.cap = None
        self.is_playing = False
        
        logger.info("VideoPlayer 초기화 완료")
    
    def play_video(self, video_path: str):
        """
        동영상 재생
        
        Args:
            video_path: 동영상 파일 경로
        """
        # 파일 존재 확인
        if not Path(video_path).exists():
            logger.error(f"동영상 파일을 찾을 수 없습니다: {video_path}")
            return
        
        logger.info(f"동영상 재생 시작: {video_path}")
        
        # 동영상 파일 열기
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            logger.error("동영상 파일을 열 수 없습니다.")
            return
        
        # 동영상 정보 가져오기
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"동영상 정보: {frame_count}프레임, {fps:.1f}fps, {duration:.1f}초")
        
        self.is_playing = True
        current_frame = 0
        start_time = time.time()
        
        logger.info("재생을 시작합니다. 'q' 키를 누르면 종료됩니다.")
        
        try:
            while self.is_playing:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("동영상 재생이 완료되었습니다.")
                    break
                
                current_frame += 1
                current_time = current_frame / fps if fps > 0 else 0
                
                # 프레임에 정보 표시
                cv2.putText(frame, f"Frame: {current_frame}/{frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {current_time:.1f}s / {duration:.1f}s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to stop", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 화면에 표시
                cv2.imshow('Video Player', frame)
                
                # 실제 FPS에 맞춰서 대기 시간 계산 (밀리초 단위)
                wait_time = int(1000 / fps) if fps > 0 else 33  # 기본 30fps
                
                # 'q' 키로 종료 또는 ESC 키로 종료
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 또는 ESC
                    logger.info("사용자에 의해 재생이 중단되었습니다.")
                    break
        
        except Exception as e:
            logger.error(f"재생 중 오류 발생: {e}")
        
        finally:
            # 리소스 해제
            self.stop_playing()
            
            # 재생 정보 출력
            total_time = time.time() - start_time
            logger.info(f"재생 완료: {total_time:.2f}초, {current_frame}프레임 재생")
    
    def stop_playing(self):
        """재생 중지 및 리소스 해제"""
        self.is_playing = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
        logger.info("리소스 해제 완료")


def main():
    """메인 함수 - 테스트용"""
    import sys
    
    if len(sys.argv) != 2:
        print("사용법: python video_player.py <동영상_파일_경로>")
        print("예시: python video_player.py posture_video_20250705_222943.mp4")
        return
    
    video_path = sys.argv[1]
    
    print("🎬 동영상 재생을 시작합니다...")
    print(f"📁 파일: {video_path}")
    print("📝 'q' 키를 누르면 재생이 종료됩니다.")
    print()
    
    player = VideoPlayer()
    player.play_video(video_path)
    
    print("✅ 재생이 완료되었습니다!")


if __name__ == "__main__":
    main() 