#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단순한 동영상 촬영 모듈

이 모듈은 웹캠을 통해 동영상을 촬영하고 파일로 저장하는 기능을 제공합니다.
실행과 동시에 촬영이 시작되고, 'q' 키를 누르면 저장 후 종료됩니다.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# 설정 파일 import
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import VIDEO_SETTINGS, PATHS


class VideoRecorder:
    """
    단순한 동영상 촬영 클래스
    
    웹캠을 통해 동영상을 촬영하고 파일로 저장합니다.
    """
    
    def __init__(self):
        """VideoRecorder 초기화"""
        self.cap = None
        self.video_writer = None
        self.is_recording = False
        
        logger.info("VideoRecorder 초기화 완료")
    
    def start_recording(self, output_path: str | None = None):
        """
        동영상 촬영 시작
        
        Args:
            output_path: 출력 파일 경로 (기본값: 자동 생성)
        """
        # 출력 파일 경로 설정
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"posture_video_{timestamp}.mp4"
        else:
            output_path = str(output_path)
        
        logger.info(f"촬영 시작: {output_path}")
        
        # 웹캠 초기화 (pose_data_collector.py 방식)
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            logger.error("웹캠을 열 수 없습니다.")
            return
        
        # 웹캠 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 실제 해상도 확인
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"웹캠 해상도: {width}x{height}")
        
        # 비디오 작성자 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        self.video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        if not self.video_writer.isOpened():
            logger.error("비디오 작성자를 열 수 없습니다.")
            self.cap.release()
            return
        
        self.is_recording = True
        frame_count = 0
        start_time = time.time()
        
        logger.info("촬영을 시작합니다. 'q' 키를 누르면 종료됩니다.")
        
        try:
            while self.is_recording:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("프레임을 읽을 수 없습니다.")
                    break
                
                # 미러링 적용
                frame = cv2.flip(frame, 1)
                
                # 프레임에 정보 표시
                elapsed_time = time.time() - start_time
                cv2.putText(frame, f"Recording: {elapsed_time:.1f}s", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {frame_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, "Press 'q' to stop", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 비디오에 프레임 저장
                self.video_writer.write(frame)
                frame_count += 1
                
                # 화면에 표시
                cv2.imshow('Video Recording', frame)
                
                # 'q' 키로 종료
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("'q' 키가 눌렸습니다. 촬영을 종료합니다.")
                    break
        
        except Exception as e:
            logger.error(f"촬영 중 오류 발생: {e}")
        
        finally:
            # 리소스 해제
            self.stop_recording()
            
            # 촬영 정보 출력
            total_time = time.time() - start_time
            logger.info(f"촬영 완료: {total_time:.2f}초, {frame_count}프레임")
            logger.info(f"저장된 파일: {output_path}")
    
    def stop_recording(self):
        """촬영 중지 및 리소스 해제"""
        self.is_recording = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
        logger.info("리소스 해제 완료")


def main():
    """메인 함수 - 실행과 동시에 촬영 시작"""
    print("🎥 동영상 촬영을 시작합니다...")
    print("📝 'q' 키를 누르면 촬영이 종료되고 파일이 저장됩니다.")
    print()
    
    recorder = VideoRecorder()
    recorder.start_recording()
    
    print("✅ 촬영이 완료되었습니다!")


if __name__ == "__main__":
    main() 