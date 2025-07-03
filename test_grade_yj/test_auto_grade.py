#!/usr/bin/env python3
"""
자동 등급 판별 기능 테스트 스크립트
"""

import sys
import os
import pickle
import numpy as np

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_auto_grade_prediction():
    """자동 등급 판별 기능 테스트"""
    print("=== 자동 등급 판별 기능 테스트 ===")
    
    # 모델 로드
    model_paths = [
        "collector/pose_grade_model.pkl",
        "ml_models/pose_grade_model.pkl",
        "pose_grade_model.pkl"
    ]
    
    model = None
    feature_names = None
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                    model = model_data['model']
                    feature_names = model_data['feature_names']
                print(f"모델 로드 성공: {path}")
                break
            except Exception as e:
                print(f"모델 로드 실패 ({path}): {e}")
    
    if model is None:
        print("모델을 로드할 수 없습니다.")
        return
    
    # 테스트 데이터 생성 (정상적인 자세)
    test_features = [
        1.5,    # neck_angle (정상)
        -0.5,   # spine_angle (정상)
        0.01,   # shoulder_asymmetry (정상)
        0.01,   # pelvic_tilt (정상)
        1,      # view_front (정면)
        0,      # view_side
        0.005,  # neck_vertical_deviation (정상)
        1.5,    # neck_angle_detailed
        -0.5,   # spine_angle_detailed
        0,      # spine_hunched (아니오)
        0.01,   # shoulder_height_diff
        179.5,  # shoulder_angle
        0,      # shoulder_asymmetric (아니오)
        0.01,   # pelvic_height_diff
        179.5,  # pelvic_angle
        0,      # pelvic_tilted (아니오)
        # 새로운 파생 특성들
        0.75,   # neck_spine_interaction
        0.02,   # total_asymmetry
        95.0,   # overall_score
        2.25,   # neck_angle_squared
        0.25,   # spine_angle_squared
        0.0001, # shoulder_asymmetry_squared
        0.0001, # pelvic_tilt_squared
        0       # problem_count
    ]
    
    print(f"\n테스트 특성:")
    for i, (name, value) in enumerate(zip(feature_names, test_features)):
        print(f"  {name}: {value}")
    
    # 예측 수행
    try:
        prediction = model.predict([test_features])[0]
        print(f"\n예측 결과: {prediction}")
        
        # 등급별 확률 확인
        probabilities = model.predict_proba([test_features])[0]
        print(f"\n등급별 확률:")
        grade_names = ['a', 'b', 'c', 'd', 'e']
        for grade, prob in zip(grade_names, probabilities):
            print(f"  {grade}: {prob:.3f}")
            
    except Exception as e:
        print(f"예측 중 오류: {e}")
    
    # 나쁜 자세 테스트
    print(f"\n=== 나쁜 자세 테스트 ===")
    bad_features = [
        15.0,   # neck_angle (나쁨)
        8.0,    # spine_angle (나쁨)
        0.08,   # shoulder_asymmetry (나쁨)
        0.08,   # pelvic_tilt (나쁨)
        1,      # view_front (정면)
        0,      # view_side
        0.02,   # neck_vertical_deviation (나쁨)
        15.0,   # neck_angle_detailed
        8.0,    # spine_angle_detailed
        1,      # spine_hunched (예)
        0.08,   # shoulder_height_diff
        175.0,  # shoulder_angle
        1,      # shoulder_asymmetric (예)
        0.08,   # pelvic_height_diff
        175.0,  # pelvic_angle
        1,      # pelvic_tilted (예)
        # 새로운 파생 특성들
        120.0,  # neck_spine_interaction
        0.16,   # total_asymmetry
        45.0,   # overall_score
        225.0,  # neck_angle_squared
        64.0,   # spine_angle_squared
        0.0064, # shoulder_asymmetry_squared
        0.0064, # pelvic_tilt_squared
        4       # problem_count
    ]
    
    try:
        bad_prediction = model.predict([bad_features])[0]
        print(f"나쁜 자세 예측 결과: {bad_prediction}")
        
        bad_probabilities = model.predict_proba([bad_features])[0]
        print(f"나쁜 자세 등급별 확률:")
        for grade, prob in zip(grade_names, bad_probabilities):
            print(f"  {grade}: {prob:.3f}")
            
    except Exception as e:
        print(f"나쁜 자세 예측 중 오류: {e}")

if __name__ == "__main__":
    test_auto_grade_prediction() 