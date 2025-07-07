#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
랜드마크 추출 모듈

MediaPipe Pose를 사용하여 프레임 이미지에서 랜드마크를 추출하고,
카메라 위치를 자동 감지하여 측면별 랜드마크를 선택합니다.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json


class LandmarkExtractor:
    """MediaPipe를 사용한 랜드마크 추출 클래스"""
    
    def __init__(self):
        """초기화"""
        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # 이미지 분석용
            model_complexity=2,      # 높은 정확도
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # 랜드마크 이름 매핑
        self.landmark_names = {
            0: 'NOSE',
            1: 'LEFT_EYE_INNER', 2: 'LEFT_EYE', 3: 'LEFT_EYE_OUTER',
            4: 'RIGHT_EYE_INNER', 5: 'RIGHT_EYE', 6: 'RIGHT_EYE_OUTER',
            7: 'LEFT_EAR', 8: 'RIGHT_EAR',
            9: 'MOUTH_LEFT', 10: 'MOUTH_RIGHT',
            11: 'LEFT_SHOULDER', 12: 'RIGHT_SHOULDER',
            13: 'LEFT_ELBOW', 14: 'RIGHT_ELBOW',
            15: 'LEFT_WRIST', 16: 'RIGHT_WRIST',
            17: 'LEFT_PINKY', 18: 'RIGHT_PINKY',
            19: 'LEFT_INDEX', 20: 'RIGHT_INDEX',
            21: 'LEFT_THUMB', 22: 'RIGHT_THUMB',
            23: 'LEFT_HIP', 24: 'RIGHT_HIP',
            25: 'LEFT_KNEE', 26: 'RIGHT_KNEE',
            27: 'LEFT_ANKLE', 28: 'RIGHT_ANKLE',
            29: 'LEFT_HEEL', 30: 'RIGHT_HEEL',
            31: 'LEFT_FOOT_INDEX', 32: 'RIGHT_FOOT_INDEX'
        }
        
        # CVA 계산용 핵심 랜드마크
        self.cva_landmarks = {
            'ear': [7, 8],      # LEFT_EAR, RIGHT_EAR
            'shoulder': [11, 12], # LEFT_SHOULDER, RIGHT_SHOULDER
            'hip': [23, 24]     # LEFT_HIP, RIGHT_HIP
        }
    
    # def detect_camera_position(self, landmarks: List) -> str:
    #     """
    #     카메라 위치 감지 (왼쪽/오른쪽 측면)
        
    #     Args:
    #         landmarks: MediaPipe 랜드마크 리스트
            
    #     Returns:
    #         'left' 또는 'right' (측면 위치)
    #     """
    #     try:
    #         # 어깨와 골반의 x좌표를 이용한 측면 감지
    #         left_shoulder_x = landmarks[11].x  # LEFT_SHOULDER
    #         right_shoulder_x = landmarks[12].x  # RIGHT_SHOULDER
    #         left_hip_x = landmarks[23].x        # LEFT_HIP
    #         right_hip_x = landmarks[24].x       # RIGHT_HIP
            
    #         # 어깨와 골반의 평균 x좌표
    #         shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
    #         hip_center_x = (left_hip_x + right_hip_x) / 2
            
    #         # 측면 판정: 어깨와 골반이 모두 화면의 한쪽에 치우쳐 있으면 측면
    #         # 왼쪽 측면: 어깨와 골반이 모두 화면의 오른쪽에 위치
    #         # 오른쪽 측면: 어깨와 골반이 모두 화면의 왼쪽에 위치
            
    #         if shoulder_center_x > 0.6 and hip_center_x > 0.6:
    #             return 'left'  # 왼쪽 측면에서 촬영
    #         elif shoulder_center_x < 0.4 and hip_center_x < 0.4:
    #             return 'right'  # 오른쪽 측면에서 촬영
    #         else:
    #             # 명확하지 않은 경우 어깨 비대칭으로 판정
    #             shoulder_diff = abs(left_shoulder_x - right_shoulder_x)
    #             if shoulder_diff > 0.1:  # 어깨가 10% 이상 차이나면 측면
    #                 if left_shoulder_x < right_shoulder_x:
    #                     return 'left'  # 왼쪽 어깨가 더 왼쪽에 있으면 왼쪽 측면
    #                 else:
    #                     return 'right'  # 오른쪽 어깨가 더 왼쪽에 있으면 오른쪽 측면
    #             else:
    #                 return 'front'  # 정면 또는 불명확
                    
    #     except Exception as e:
    #         print(f"카메라 위치 감지 오류: {e}")
    #         return 'unknown'
    
    def extract_landmarks_from_image(self, image_path: str, subject_id: str = "P1") -> Optional[Dict]:
        """
        단일 이미지에서 랜드마크 추출
        
        Args:
            image_path: 이미지 파일 경로
            subject_id: 피사체 ID
            
        Returns:
            랜드마크 데이터 딕셔너리 또는 None (실패 시)
        """
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 로드할 수 없습니다: {image_path}")
                return None
                
            # BGR to RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe로 포즈 추론
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                print(f"포즈를 감지할 수 없습니다: {image_path}")
                return None
            
            # 카메라 위치는 4단계 좌표 필터링에서 결정 (현재는 빈 값)
            camera_position = ""
            
            # 랜드마크 데이터 추출
            landmarks_data = []
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append({
                    'landmark_id': i,
                    'landmark_name': self.landmark_names.get(i, f'UNKNOWN_{i}'),
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # 프레임명 추출
            frame_name = Path(image_path).stem
            
            # 결과 데이터
            result = {
                'subject_id': subject_id,
                'frame_name': frame_name,
                'image_path': image_path,
                'image_size': {
                    'width': image.shape[1],
                    'height': image.shape[0]
                },
                'camera_position': camera_position,
                'landmarks': landmarks_data,
                'extraction_time': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {image_path}, 오류: {e}")
            return None
    
    def extract_landmarks_from_directory(self, frames_dir: str, output_dir: str = None, 
                                       subject_id: str = "P1") -> List[Dict]:
        """
        디렉토리의 모든 프레임에서 랜드마크 추출
        
        Args:
            frames_dir: 프레임 이미지 디렉토리
            output_dir: 출력 디렉토리 (기본값: 자동 생성)
            subject_id: 피사체 ID
            
        Returns:
            추출된 랜드마크 데이터 리스트
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            print(f"프레임 디렉토리를 찾을 수 없습니다: {frames_dir}")
            return []
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = Path("data/landmarks")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(frames_path.glob(f"*{ext}"))
            image_files.extend(frames_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"이미지 파일을 찾을 수 없습니다: {frames_dir}")
            return []
        
        # 파일명으로 정렬
        image_files.sort(key=lambda x: x.name)
        
        print(f"🔄 {len(image_files)}개의 프레임에서 랜드마크를 추출합니다...")
        
        extracted_data = []
        successful_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"📸 처리 중: {image_file.name} ({i}/{len(image_files)})")
            
            result = self.extract_landmarks_from_image(str(image_file), subject_id)
            if result:
                extracted_data.append(result)
                successful_count += 1
            else:
                print(f"❌ 실패: {image_file.name}")
        
        print(f"✅ 랜드마크 추출 완료: {successful_count}/{len(image_files)} 성공")
        
        # CSV 파일로 저장
        if extracted_data:
            self.save_to_csv(extracted_data, output_dir)
            self.save_to_json(extracted_data, output_dir)
        
        return extracted_data
    
    def save_to_csv(self, data: List[Dict], output_dir: Path):
        """랜드마크 데이터를 CSV 파일로 저장"""
        try:
            # CSV 데이터 준비
            csv_rows = []
            
            for item in data:
                subject_id = item['subject_id']
                frame_name = item['frame_name']
                camera_position = item['camera_position']
                
                for landmark in item['landmarks']:
                    row = {
                        'subject_id': subject_id,
                        'frame_name': frame_name,
                        'camera_position': camera_position,
                        'landmark_id': landmark['landmark_id'],
                        'landmark_name': landmark['landmark_name'],
                        'x': landmark['x'],
                        'y': landmark['y'],
                        'z': landmark['z'],
                        'visibility': landmark['visibility']
                    }
                    csv_rows.append(row)
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame(csv_rows)
            csv_path = output_dir / "raw_landmarks.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f"📄 CSV 파일 저장 완료: {csv_path}")
            
        except Exception as e:
            print(f"CSV 저장 오류: {e}")
    
    def save_to_json(self, data: List[Dict], output_dir: Path):
        """랜드마크 데이터를 JSON 파일로 저장"""
        try:
            json_path = output_dir / "raw_landmarks.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"📄 JSON 파일 저장 완료: {json_path}")
            
        except Exception as e:
            print(f"JSON 저장 오류: {e}")
    
    def get_cva_landmarks(self, landmarks_data: List[Dict], camera_position: str) -> Dict:
        """
        CVA 계산용 핵심 랜드마크 추출
        
        Args:
            landmarks_data: 랜드마크 데이터
            camera_position: 카메라 위치 ('left', 'right', 'front')
            
        Returns:
            CVA 계산용 랜드마크 딕셔너리
        """
        cva_landmarks = {}
        
        # 랜드마크를 ID로 매핑
        landmarks_by_id = {lm['landmark_id']: lm for lm in landmarks_data}
        
        # 카메라 위치에 따른 측면별 랜드마크 선택
        if camera_position == 'left':
            # 왼쪽 측면: 왼쪽 랜드마크 사용
            cva_landmarks['ear'] = landmarks_by_id.get(7)  # LEFT_EAR
            cva_landmarks['shoulder'] = landmarks_by_id.get(11)  # LEFT_SHOULDER
            cva_landmarks['hip'] = landmarks_by_id.get(23)  # LEFT_HIP
        elif camera_position == 'right':
            # 오른쪽 측면: 오른쪽 랜드마크 사용
            cva_landmarks['ear'] = landmarks_by_id.get(8)  # RIGHT_EAR
            cva_landmarks['shoulder'] = landmarks_by_id.get(12)  # RIGHT_SHOULDER
            cva_landmarks['hip'] = landmarks_by_id.get(24)  # RIGHT_HIP
        else:
            # 정면 또는 불명확: 양쪽 평균 사용
            left_ear = landmarks_by_id.get(7)
            right_ear = landmarks_by_id.get(8)
            left_shoulder = landmarks_by_id.get(11)
            right_shoulder = landmarks_by_id.get(12)
            left_hip = landmarks_by_id.get(23)
            right_hip = landmarks_by_id.get(24)
            
            if all([left_ear, right_ear]):
                cva_landmarks['ear'] = {
                    'x': (left_ear['x'] + right_ear['x']) / 2,
                    'y': (left_ear['y'] + right_ear['y']) / 2,
                    'z': (left_ear['z'] + right_ear['z']) / 2
                }
            
            if all([left_shoulder, right_shoulder]):
                cva_landmarks['shoulder'] = {
                    'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                    'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
                    'z': (left_shoulder['z'] + right_shoulder['z']) / 2
                }
            
            if all([left_hip, right_hip]):
                cva_landmarks['hip'] = {
                    'x': (left_hip['x'] + right_hip['x']) / 2,
                    'y': (left_hip['y'] + right_hip['y']) / 2,
                    'z': (left_hip['z'] + right_hip['z']) / 2
                }
        
        return cva_landmarks
    
    def __del__(self):
        """소멸자: MediaPipe 리소스 해제"""
        if hasattr(self, 'pose'):
            self.pose.close() 