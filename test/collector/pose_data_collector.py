import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import time

# DB 초기화
conn = sqlite3.connect('pose_landmarks.db')
c = conn.cursor()

# 기존 테이블이 있다면 person_id 컬럼 추가
try:
    c.execute("ALTER TABLE pose_landmarks ADD COLUMN person_id TEXT DEFAULT 'P1'")
    print("person_id 컬럼이 추가되었습니다.")
except sqlite3.OperationalError:
    print("person_id 컬럼이 이미 존재합니다.")

# 테이블이 없다면 새로 생성
columns = ', '.join([f'landmark_{i}_x REAL, landmark_{i}_y REAL' for i in range(33)])
c.execute(f'''
    CREATE TABLE IF NOT EXISTS pose_landmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        label INTEGER,
        person_id TEXT,
        {columns}
    )
''')
conn.commit()

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
label = 3  # 기본값: 3(제외)
person_id = "P1"  # 기본값: P1

frame_count = 0

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
            print(f"라벨 변경: {label} (측면)")
        elif key == ord('3'):
            label = 3
            print(f"라벨 변경: {label} (제외)")
        # 사람 식별자 변경 (P1~P9)
        elif key == ord('p') or key == ord('P'):
            # 다음 사람으로 변경
            current_num = int(person_id[1]) if len(person_id) > 1 else 1
            next_num = (current_num % 9) + 1
            person_id = f"P{next_num}"
            print(f"사람 식별자 변경: {person_id}")
        elif key == ord('0'):
            # P0으로 설정 (테스트용)
            person_id = "P0"
            print(f"사람 식별자 변경: {person_id}")
        elif key == 27:  # ESC
            break

        # 결과 표시를 위해 다시 BGR로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 현재 라벨과 사람 식별자를 화면에 표시
        label_text = f"현재 라벨: {label}"
        if label == 1:
            label_text += " (정면)"
        elif label == 2:
            label_text += " (측면)"
        else:
            label_text += " (제외)"
        
        person_text = f"사람: {person_id}"
        
        cv2.putText(image, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, person_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, "1: 정면, 2: 측면, 3: 제외, P: 사람변경, ESC: 종료", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 10프레임마다 저장
        if results.pose_landmarks and frame_count % 10 == 0:
            row = [time.time(), label, person_id]
            for lm in results.pose_landmarks.landmark:
                # visibility가 0.5 미만이거나 x, y가 비정상적인 경우 -1로 저장
                if hasattr(lm, 'visibility') and lm.visibility < 0.5:
                    row.extend([-1, -1])
                else:
                    row.extend([lm.x, lm.y])
            c.execute(f'''
                INSERT INTO pose_landmarks (timestamp, label, person_id, {', '.join([f'landmark_{i}_x, landmark_{i}_y' for i in range(33)])})
                VALUES ({', '.join(['?'] * (3 + 33*2))})
            ''', row)
            conn.commit()
            print(f"데이터 저장 완료 - 사람: {person_id}, 라벨: {label}, 프레임: {frame_count}")

        frame_count += 1
        cv2.imshow('Pose Collector', image)

    cap.release()
    cv2.destroyAllWindows()
    conn.close()