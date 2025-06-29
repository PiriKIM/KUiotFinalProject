import sqlite3
import pandas as pd

# 데이터베이스 연결
conn = sqlite3.connect('pose_landmarks.db')

# 테이블 목록 확인
print("=== 테이블 목록 ===")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(f"- {table[0]}")

# pose_landmarks 테이블 정보 확인
print("\n=== pose_landmarks 테이블 정보 ===")
cursor.execute("PRAGMA table_info(pose_landmarks);")
columns = cursor.fetchall()
print(f"총 컬럼 수: {len(columns)}")
print("컬럼 목록:")
for col in columns:
    print(f"  - {col[1]} ({col[2]})")

# 전체 레코드 수 확인
print("\n=== 데이터 통계 ===")
cursor.execute("SELECT COUNT(*) FROM pose_landmarks;")
total_records = cursor.fetchone()[0]
print(f"총 레코드 수: {total_records}")

# 라벨별 분포 확인
print("\n=== 라벨별 분포 ===")
cursor.execute("SELECT label, COUNT(*) FROM pose_landmarks GROUP BY label;")
label_counts = cursor.fetchall()
for label, count in label_counts:
    if label == 1:
        label_name = "정면"
    elif label == 2:
        label_name = "측면"
    else:
        label_name = "제외"
    print(f"라벨 {label} ({label_name}): {count}개")

# 최근 5개 레코드 확인
print("\n=== 최근 5개 레코드 ===")
cursor.execute("SELECT id, timestamp, label FROM pose_landmarks ORDER BY id DESC LIMIT 5;")
recent_records = cursor.fetchall()
for record in recent_records:
    id_val, timestamp, label = record
    if label == 1:
        label_name = "정면"
    elif label == 2:
        label_name = "측면"
    else:
        label_name = "제외"
    print(f"ID: {id_val}, 시간: {timestamp:.2f}, 라벨: {label} ({label_name})")

# -1 값이 있는지 확인 (감지되지 않은 관절)
print("\n=== 감지되지 않은 관절 확인 ===")
cursor.execute("SELECT COUNT(*) FROM pose_landmarks WHERE landmark_0_x = -1;")
undetected_count = cursor.fetchone()[0]
print(f"감지되지 않은 관절이 포함된 레코드 수: {undetected_count}")

# 첫 번째 레코드의 상세 정보 (랜드마크 좌표 일부)
print("\n=== 첫 번째 레코드 상세 정보 (랜드마크 0-4번) ===")
cursor.execute("SELECT id, label, landmark_0_x, landmark_0_y, landmark_1_x, landmark_1_y, landmark_2_x, landmark_2_y, landmark_3_x, landmark_3_y, landmark_4_x, landmark_4_y FROM pose_landmarks LIMIT 1;")
first_record = cursor.fetchone()
if first_record:
    print(f"ID: {first_record[0]}, 라벨: {first_record[1]}")
    for i in range(5):
        x_idx = 2 + i * 2
        y_idx = 2 + i * 2 + 1
        x_val = first_record[x_idx]
        y_val = first_record[y_idx]
        print(f"  랜드마크 {i}: ({x_val}, {y_val})")

conn.close()
print("\n=== 데이터베이스 확인 완료 ===") 