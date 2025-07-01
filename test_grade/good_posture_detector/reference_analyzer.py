import cv2
import mediapipe as mp
import numpy as np
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

class ReferenceAnalyzer:
    """
    기준 이미지들에서 MediaPipe를 사용하여 랜드마크를 추출하고 분석하는 클래스
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # 이미지 분석용
            model_complexity=2,      # 높은 정확도
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # 결과 저장용 디렉토리
        self.output_dir = Path("data/reference_landmarks")
        self.output_dir.mkdir(exist_ok=True)
        
        # 분석 결과 저장
        self.good_posture_samples = []
        self.bad_posture_samples = []
        
    def extract_landmarks_from_image(self, image_path: str) -> Optional[Dict]:
        """
        단일 이미지에서 랜드마크 추출
        
        Args:
            image_path: 이미지 파일 경로
            
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
            
            # 랜드마크 데이터 추출
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            
            # 기본 정보
            image_info = {
                'image_path': image_path,
                'image_size': {
                    'width': image.shape[1],
                    'height': image.shape[0]
                },
                'landmarks': landmarks,
                'extraction_time': datetime.now().isoformat()
            }
            
            return image_info
            
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {image_path}, 오류: {e}")
            return None
    
    def analyze_posture_features(self, landmarks: List[Dict]) -> Dict:
        """
        랜드마크에서 자세 특징 추출
        
        Args:
            landmarks: MediaPipe 랜드마크 리스트
            
        Returns:
            자세 특징 딕셔너리
        """
        try:
            # MediaPipe enum을 사용한 랜드마크 인덱스
            LEFT_EAR = self.mp_pose.PoseLandmark.LEFT_EAR.value
            RIGHT_EAR = self.mp_pose.PoseLandmark.RIGHT_EAR.value
            LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP.value
            RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP.value
            LEFT_KNEE = self.mp_pose.PoseLandmark.LEFT_KNEE.value
            RIGHT_KNEE = self.mp_pose.PoseLandmark.RIGHT_KNEE.value
            
            # 랜드마크 좌표 추출
            left_ear = landmarks[LEFT_EAR]
            right_ear = landmarks[RIGHT_EAR]
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            left_hip = landmarks[LEFT_HIP]
            right_hip = landmarks[RIGHT_HIP]
            left_knee = landmarks[LEFT_KNEE]
            right_knee = landmarks[RIGHT_KNEE]
            
            # 1. 목 정렬도 계산 (귀-어깨 수직 정렬)
            ear_center_x = (left_ear['x'] + right_ear['x']) / 2
            ear_center_y = (left_ear['y'] + right_ear['y']) / 2
            shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            neck_alignment = abs(ear_center_x - shoulder_center_x)
            
            # 2. 척추 직선성 계산 (어깨-골반 수직 정렬)
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            hip_center_y = (left_hip['y'] + right_hip['y']) / 2
            
            spine_alignment = abs(shoulder_center_x - hip_center_x)
            
            # 3. 어깨 대칭성 계산
            shoulder_height_diff = abs(left_shoulder['y'] - right_shoulder['y'])
            
            # 4. 골반 대칭성 계산
            hip_height_diff = abs(left_hip['y'] - right_hip['y'])
            
            # 5. 목 각도 계산
            neck_angle = self._calculate_neck_angle(landmarks)
            
            # 6. 척추 각도 계산
            spine_angle = self._calculate_spine_angle(landmarks)
            
            # 특징 정리
            features = {
                'neck_alignment': neck_alignment,
                'spine_alignment': spine_alignment,
                'shoulder_symmetry': shoulder_height_diff,
                'pelvic_symmetry': hip_height_diff,
                'neck_angle': neck_angle,
                'spine_angle': spine_angle,
                'overall_balance': self._calculate_overall_balance(landmarks)
            }
            
            return features
            
        except Exception as e:
            print(f"특징 추출 중 오류 발생: {e}")
            return {}
    
    def _calculate_neck_angle(self, landmarks: List[Dict]) -> float:
        """목 각도 계산"""
        try:
            # MediaPipe enum을 사용한 랜드마크 인덱스
            LEFT_EAR = self.mp_pose.PoseLandmark.LEFT_EAR.value
            RIGHT_EAR = self.mp_pose.PoseLandmark.RIGHT_EAR.value
            LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            
            left_ear = landmarks[LEFT_EAR]
            right_ear = landmarks[RIGHT_EAR]
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            
            # 귀 중심점
            ear_center = np.array([(left_ear['x'] + right_ear['x']) / 2, 
                                  (left_ear['y'] + right_ear['y']) / 2])
            
            # 어깨 중심점
            shoulder_center = np.array([(left_shoulder['x'] + right_shoulder['x']) / 2,
                                       (left_shoulder['y'] + right_shoulder['y']) / 2])
            
            # 수직선 기준점
            vertical_point = np.array([shoulder_center[0], ear_center[1]])
            
            # 각도 계산
            angle = self._calculate_angle(ear_center, shoulder_center, vertical_point)
            return angle
            
        except Exception as e:
            print(f"목 각도 계산 오류: {e}")
            return 0.0
    
    def _calculate_spine_angle(self, landmarks: List[Dict]) -> float:
        """척추 각도 계산"""
        try:
            # MediaPipe enum을 사용한 랜드마크 인덱스
            LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP.value
            RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP.value
            
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            left_hip = landmarks[LEFT_HIP]
            right_hip = landmarks[RIGHT_HIP]
            
            # 어깨 중심점
            shoulder_center = np.array([(left_shoulder['x'] + right_shoulder['x']) / 2,
                                       (left_shoulder['y'] + right_shoulder['y']) / 2])
            
            # 골반 중심점
            hip_center = np.array([(left_hip['x'] + right_hip['x']) / 2,
                                  (left_hip['y'] + right_hip['y']) / 2])
            
            # 수직선 기준점
            vertical_point = np.array([shoulder_center[0], hip_center[1]])
            
            # 각도 계산
            angle = self._calculate_angle(shoulder_center, hip_center, vertical_point)
            return angle
            
        except Exception as e:
            print(f"척추 각도 계산 오류: {e}")
            return 0.0
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """세 점으로 각도 계산"""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _calculate_overall_balance(self, landmarks: List[Dict]) -> float:
        """전체 균형 점수 계산 (0~1, 높을수록 좋음)"""
        try:
            # MediaPipe enum을 사용한 랜드마크 인덱스
            LEFT_EAR = self.mp_pose.PoseLandmark.LEFT_EAR.value
            LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP.value
            RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP.value
            
            # 각 특징의 점수를 0~1로 정규화
            neck_score = max(0, 1 - abs(landmarks[LEFT_EAR]['x'] - landmarks[LEFT_SHOULDER]['x']) * 10)
            spine_score = max(0, 1 - abs(landmarks[LEFT_SHOULDER]['x'] - landmarks[LEFT_HIP]['x']) * 10)
            shoulder_score = max(0, 1 - abs(landmarks[LEFT_SHOULDER]['y'] - landmarks[RIGHT_SHOULDER]['y']) * 20)
            hip_score = max(0, 1 - abs(landmarks[LEFT_HIP]['y'] - landmarks[RIGHT_HIP]['y']) * 20)
            
            # 가중 평균
            overall_score = (neck_score * 0.3 + spine_score * 0.3 + 
                           shoulder_score * 0.2 + hip_score * 0.2)
            
            return overall_score
            
        except Exception as e:
            print(f"전체 균형 점수 계산 오류: {e}")
            return 0.0
    
    def analyze_reference_images(self, image_folder: str, posture_type: str = "unknown"):
        """
        폴더 내 모든 이미지 분석
        
        Args:
            image_folder: 이미지 폴더 경로
            posture_type: 자세 타입 ("good" 또는 "bad")
        """
        print(f"\n=== {posture_type.upper()} 자세 이미지 분석 시작 ===")
        print(f"폴더: {image_folder}")
        
        # 이미지 파일 찾기
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
        
        if not image_files:
            print(f"이미지 파일을 찾을 수 없습니다: {image_folder}")
            return
        
        print(f"발견된 이미지: {len(image_files)}개")
        
        # 각 이미지 분석
        successful_analyses = 0
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 분석 중: {os.path.basename(image_path)}")
            
            # 랜드마크 추출
            landmark_data = self.extract_landmarks_from_image(image_path)
            if landmark_data is None:
                continue
            
            # 자세 특징 분석
            features = self.analyze_posture_features(landmark_data['landmarks'])
            landmark_data['posture_type'] = posture_type
            landmark_data['features'] = features
            
            # 결과 저장
            if posture_type == "good":
                self.good_posture_samples.append(landmark_data)
            elif posture_type == "bad":
                self.bad_posture_samples.append(landmark_data)
            
            successful_analyses += 1
            print(f"  ✓ 성공 (목 정렬: {features.get('neck_alignment', 0):.3f}, "
                  f"어깨 대칭: {features.get('shoulder_symmetry', 0):.3f})")
        
        print(f"\n=== 분석 완료 ===")
        print(f"성공: {successful_analyses}/{len(image_files)}")
        
        # 결과 저장
        self.save_analysis_results(posture_type)
    
    def save_analysis_results(self, posture_type: str):
        """분석 결과를 JSON 파일로 저장"""
        if posture_type == "good":
            samples = self.good_posture_samples
        elif posture_type == "bad":
            samples = self.bad_posture_samples
        else:
            return
        
        if not samples:
            print(f"저장할 {posture_type} 자세 샘플이 없습니다.")
            return
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{posture_type}_posture_landmarks_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # JSON으로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"결과 저장: {filepath}")
        print(f"저장된 샘플 수: {len(samples)}개")
    
    def generate_summary_report(self):
        """분석 결과 요약 리포트 생성"""
        print("\n" + "="*60)
        print("📊 자세 분석 결과 요약")
        print("="*60)
        
        # 좋은 자세 통계
        if self.good_posture_samples:
            good_features = [sample['features'] for sample in self.good_posture_samples]
            print(f"\n✅ 좋은 자세 샘플: {len(self.good_posture_samples)}개")
            self._print_feature_statistics(good_features, "좋은 자세")
        
        # 나쁜 자세 통계
        if self.bad_posture_samples:
            bad_features = [sample['features'] for sample in self.bad_posture_samples]
            print(f"\n❌ 나쁜 자세 샘플: {len(self.bad_posture_samples)}개")
            self._print_feature_statistics(bad_features, "나쁜 자세")
        
        # 비교 분석
        if self.good_posture_samples and self.bad_posture_samples:
            print(f"\n📈 비교 분석:")
            self._compare_posture_types()
    
    def _print_feature_statistics(self, features_list: List[Dict], posture_type: str):
        """특징 통계 출력"""
        if not features_list:
            return
        
        feature_names = ['neck_alignment', 'spine_alignment', 'shoulder_symmetry', 
                        'pelvic_symmetry', 'neck_angle', 'spine_angle', 'overall_balance']
        
        for feature in feature_names:
            values = [f[feature] for f in features_list if feature in f]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {feature}: 평균 {mean_val:.3f} ± {std_val:.3f}")
    
    def _compare_posture_types(self):
        """좋은 자세와 나쁜 자세 비교"""
        good_features = [sample['features'] for sample in self.good_posture_samples]
        bad_features = [sample['features'] for sample in self.bad_posture_samples]
        
        feature_names = ['neck_alignment', 'spine_alignment', 'shoulder_symmetry', 
                        'pelvic_symmetry', 'overall_balance']
        
        for feature in feature_names:
            good_values = [f[feature] for f in good_features if feature in f]
            bad_values = [f[feature] for f in bad_features if feature in f]
            
            if good_values and bad_values:
                good_mean = np.mean(good_values)
                bad_mean = np.mean(bad_values)
                difference = good_mean - bad_mean
                
                print(f"  {feature}: 좋은 자세({good_mean:.3f}) vs 나쁜 자세({bad_mean:.3f}) "
                      f"차이: {difference:.3f}")
    
    def cleanup(self):
        """리소스 정리"""
        self.pose.close()


def main():
    """메인 실행 함수"""
    analyzer = ReferenceAnalyzer()
    
    try:
        # 좋은 자세 이미지 분석
        good_posture_folder = "good_posture_detector/data/good_posture_samples"
        if os.path.exists(good_posture_folder):
            analyzer.analyze_reference_images(good_posture_folder, "good")
        
        # 나쁜 자세 이미지 분석
        bad_posture_folder = "good_posture_detector/data/bad_posture_samples"
        if os.path.exists(bad_posture_folder):
            analyzer.analyze_reference_images(bad_posture_folder, "bad")
        
        # 요약 리포트 생성
        analyzer.generate_summary_report()
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
    
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
