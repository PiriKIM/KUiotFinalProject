import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

# 저장 디렉토리 생성
save_dir = Path("/home/piri/KUiotFinalProject/merge_mp/data/")
save_dir.mkdir(parents=True, exist_ok=True)

# CSV 파일 경로
csv_file = save_dir / "pose_landmarks_P1_additional.csv"

# 기존 CSV 파일이 있으면 로드, 없으면 새로 생성
if csv_file.exists():
    df = pd.read_csv(csv_file)
    print(f"기존 CSV 파일을 로드했습니다: {csv_file}")
    print(f"기존 데이터 수: {len(df)}")
else:
    # 새로운 DataFrame 생성
    columns = ['id', 'timestamp', 'label', 'person_id']
    for i in range(33):
        columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
    
    df = pd.DataFrame(columns=columns)
    print(f"새로운 CSV 파일을 생성합니다: {csv_file}")

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
label = 0  # 기본값: 0(라벨링 모드)
person_id = "P1"  # 기본값: P1

frame_count = 0
data_count = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 키보드 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            label = 1
            print(f"라벨 변경: {label} (정면)")
        elif key == ord('2'):
            label = 2
            print(f"라벨 변경: {label} (왼쪽 측면)")
        elif key == ord('3'):
            label = 3
            print(f"라벨 변경: {label} (오른쪽 측면)")
        elif key == ord('0'):
            label = 0
            print(f"라벨 변경: {label} (기타)")
        # 사람 식별자 변경 (P1~P9)
        elif key == ord('p') or key == ord('P'):
            # 다음 사람으로 변경
            current_num = int(person_id[1]) if len(person_id) > 1 else 1
            next_num = (current_num % 9) + 1
            person_id = f"P{next_num}"
            print(f"사람 식별자 변경: {person_id}")

        elif key == 27:  # ESC
            break

        # 결과 표시를 위해 다시 BGR로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 현재 라벨과 사람 식별자를 화면에 표시
        label_text = f"현재 라벨: {label}"
        if label == 0:
            label_text += " (기타)"
        elif label == 1:
            label_text += " (정면)"
        elif label == 2:
            label_text += " (왼쪽 측면)"
        elif label == 3:
            label_text += " (오른쪽 측면)"
        
        person_text = f"사람: {person_id}"
        
        cv2.putText(image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, person_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, "0: 기타, 1: 정면, 2: 왼쪽측면, 3: 오른쪽측면", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "P: 사람변경, ESC: 종료", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 10프레임마다 저장
        if results.pose_landmarks and frame_count % 10 == 0:
            # 새로운 행 데이터 생성
            new_row = {
                'id': len(df) + 1,
                'timestamp': time.time(),
                'label': label,
                'person_id': person_id
            }
            
            # 랜드마크 데이터 추가
            for i, lm in enumerate(results.pose_landmarks.landmark):
                # visibility가 0.5 미만이거나 x, y가 비정상적인 경우 -1로 저장
                if hasattr(lm, 'visibility') and lm.visibility < 0.5:
                    new_row[f'landmark_{i}_x'] = -1
                    new_row[f'landmark_{i}_y'] = -1
                else:
                    new_row[f'landmark_{i}_x'] = lm.x
                    new_row[f'landmark_{i}_y'] = lm.y
            
            # DataFrame에 새 행 추가
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # CSV 파일에 저장
            df.to_csv(csv_file, index=False)
            
            data_count += 1
            print(f"데이터 저장 완료 - 사람: {person_id}, 라벨: {label}, 프레임: {frame_count}, 총 데이터: {len(df)}")

        frame_count += 1
        cv2.imshow('Pose Collector 4-Way', image)

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== 수집 완료 ===")
    print(f"저장 위치: {csv_file}")
    print(f"총 수집된 데이터: {len(df)}")
    print(f"라벨 분포:")
    print(df['label'].value_counts().sort_index())
    print(f"사람별 분포:")
    print(df['person_id'].value_counts().sort_index()) 