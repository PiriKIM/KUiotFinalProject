import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import os
from datetime import datetime

# DB 경로 및 이미지 폴더 경로
DB_PATH = "../pose_grade_data.db"
FOLDERS = [
    ("../data/good_posture_samples", 'a'),  # A등급
    ("../data/bad_posture_samples", 'd'),   # D등급
]

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# DB 연결
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

total_success = 0
total_fail = 0

for folder, grade in FOLDERS:
    if not os.path.exists(folder):
        print(f"[경고] 폴더 없음: {folder}")
        continue
    img_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n[{folder}] 총 {len(img_files)}개 이미지 파일 발견 (등급: {grade.upper()})")
    success_count = 0
    fail_count = 0
    for img_file in img_files:
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[실패] 이미지 로드 실패: {img_file}")
            fail_count += 1
            continue
        # MediaPipe로 랜드마크 추출
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            print(f"[실패] 랜드마크 미검출: {img_file}")
            fail_count += 1
            continue
        landmarks = results.pose_landmarks.landmark
        # 66개 랜드마크 값 (visibility < 0.5면 -1)
        landmark_values = []
        for lm in landmarks:
            if hasattr(lm, 'visibility') and lm.visibility < 0.5:
                landmark_values.extend([-1, -1])
            else:
                landmark_values.extend([lm.x, lm.y])
        # 66개 컬럼명
        landmark_columns = []
        for i in range(33):
            landmark_columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
        # 분석 결과 (간단히 0/None)
        neck_angle = 0
        spine_angle = 0
        shoulder_asymmetry = 0
        pelvic_tilt = 0
        total_score = None
        analysis_results = None
        # participant_id: 파일명(확장자 제외)
        participant_id = os.path.splitext(img_file)[0]
        # 컬럼 및 값
        columns = ['timestamp', 'participant_id', 'view_angle', 'pose_grade', 'auto_grade',
                   'neck_angle', 'spine_angle', 'shoulder_asymmetry', 'pelvic_tilt',
                   'total_score', 'analysis_results'] + landmark_columns
        placeholders = ['?'] * len(columns)
        values = [
            datetime.now().isoformat(),
            participant_id,
            '2',  # view_angle: 측면
            grade,  # pose_grade: 폴더별 등급
            grade,  # auto_grade: 폴더별 등급
            neck_angle,
            spine_angle,
            shoulder_asymmetry,
            pelvic_tilt,
            total_score,
            analysis_results
        ] + landmark_values
        try:
            cursor.execute(f"""
                INSERT INTO pose_grade_data
                ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """, values)
            conn.commit()
            print(f"[성공] {img_file} → DB 저장 완료 (등급: {grade.upper()})")
            success_count += 1
        except Exception as e:
            print(f"[실패] DB 저장 오류: {img_file} ({e})")
            fail_count += 1
    print(f"[{folder}] 성공: {success_count}개, 실패: {fail_count}개")
    total_success += success_count
    total_fail += fail_count

conn.close()
print(f"\n=== 전체 요약 ===")
print(f"성공: {total_success}개, 실패: {total_fail}개") 