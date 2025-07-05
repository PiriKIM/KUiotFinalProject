#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프레임 분할 모듈

이 모듈은 동영상을 등간격으로 50개 프레임으로 분할하여 저장하는 기능을 제공합니다.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    동영상 프레임 분할 클래스
    
    동영상을 등간격으로 50개 프레임으로 분할하여 저장합니다.
    """
    
    def __init__(self):
        """FrameExtractor 초기화"""
        self.cap = None
        
        logger.info("FrameExtractor 초기화 완료")
    
    def extract_frames(self, video_path: str, output_dir: str | None = None, num_frames: int = 50) -> List[str]:
        """
        동영상에서 프레임 추출
        
        Args:
            video_path: 동영상 파일 경로
            output_dir: 출력 디렉토리 (기본값: 자동 생성)
            num_frames: 추출할 프레임 수 (기본값: 50)
            
        Returns:
            저장된 프레임 파일 경로 리스트
        """
        # 파일 존재 확인
        if not Path(video_path).exists():
            logger.error(f"동영상 파일을 찾을 수 없습니다: {video_path}")
            return []
        
        # 출력 디렉토리 설정
        if output_dir is None:
            video_name = Path(video_path).stem
            output_dir = f"frames_{video_name}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"프레임 추출 시작: {video_path}")
        logger.info(f"출력 디렉토리: {output_path}")
        logger.info(f"추출할 프레임 수: {num_frames}")
        
        # 동영상 파일 열기
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            logger.error("동영상 파일을 열 수 없습니다.")
            return []
        
        # 동영상 정보 가져오기
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"동영상 정보: {total_frames}프레임, {fps:.1f}fps, {duration:.1f}초")
        
        # 프레임 간격 계산
        if total_frames <= num_frames:
            # 총 프레임이 요청한 프레임 수보다 적으면 모든 프레임 추출
            frame_indices = list(range(total_frames))
            logger.info(f"총 프레임이 {num_frames}개보다 적어서 모든 프레임을 추출합니다.")
        else:
            # 등간격으로 프레임 인덱스 계산
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            # 마지막 프레임이 중복되지 않도록 조정
            if frame_indices[-1] >= total_frames:
                frame_indices[-1] = total_frames - 1
        
        logger.info(f"추출할 프레임 인덱스: {frame_indices[:5]}...{frame_indices[-5:]}")
        
        # 프레임 추출 및 저장
        saved_frames = []
        current_frame = 0
        
        try:
            for i, frame_index in enumerate(frame_indices):
                # 해당 프레임으로 이동
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"프레임 {frame_index}를 읽을 수 없습니다.")
                    continue
                
                # 프레임 번호 포맷팅 (01, 02, ..., 50)
                frame_number = i + 1
                frame_filename = f"frame_{frame_number:02d}.jpg"
                frame_path = output_path / frame_filename
                
                # 프레임 저장
                success = cv2.imwrite(str(frame_path), frame)
                if success:
                    saved_frames.append(str(frame_path))
                    logger.info(f"프레임 저장 완료: {frame_filename} (원본 프레임: {frame_index})")
                else:
                    logger.error(f"프레임 저장 실패: {frame_filename}")
                
                current_frame += 1
        
        except Exception as e:
            logger.error(f"프레임 추출 중 오류 발생: {e}")
        
        finally:
            # 리소스 해제
            self.release()
        
        logger.info(f"프레임 추출 완료: {len(saved_frames)}개 프레임 저장")
        return saved_frames
    
    def get_video_info(self, video_path: str) -> Tuple[int, float, float]:
        """
        동영상 정보 가져오기
        
        Args:
            video_path: 동영상 파일 경로
            
        Returns:
            (총 프레임 수, FPS, 재생 시간) 튜플
        """
        if not Path(video_path).exists():
            logger.error(f"동영상 파일을 찾을 수 없습니다: {video_path}")
            return (0, 0, 0)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("동영상 파일을 열 수 없습니다.")
            return (0, 0, 0)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return (total_frames, fps, duration)
    
    def release(self):
        """리소스 해제"""
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("리소스 해제 완료")


def main():
    """메인 함수 - 테스트용"""
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python frame_extractor.py <동영상_파일_경로> [출력_디렉토리] [프레임_수]")
        print("예시: python frame_extractor.py video.mp4 frames 50")
        return
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    num_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    print("🖼️ 프레임 추출을 시작합니다...")
    print(f"📁 동영상: {video_path}")
    print(f"📂 출력: {output_dir or '자동 생성'}")
    print(f"🎯 프레임 수: {num_frames}")
    print()
    
    extractor = FrameExtractor()
    
    # 동영상 정보 출력
    total_frames, fps, duration = extractor.get_video_info(video_path)
    print(f"📊 동영상 정보:")
    print(f"  - 총 프레임: {total_frames}")
    print(f"  - FPS: {fps:.1f}")
    print(f"  - 재생 시간: {duration:.1f}초")
    print()
    
    # 프레임 추출
    saved_frames = extractor.extract_frames(video_path, output_dir, num_frames)
    
    if saved_frames:
        print(f"✅ 프레임 추출 완료!")
        print(f"📁 저장된 프레임: {len(saved_frames)}개")
        print(f"📂 저장 위치: {Path(saved_frames[0]).parent}")
    else:
        print("❌ 프레임 추출 실패")


if __name__ == "__main__":
    main() 