import sqlite3
import pandas as pd
import os

# 저장 경로 지정
csv_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(csv_dir, exist_ok=True)

# 데이터베이스 연결
conn = sqlite3.connect('pose_landmarks.db')

# 전체 데이터를 DataFrame으로 읽기
print("데이터베이스에서 데이터를 읽는 중...")
df = pd.read_sql_query("SELECT * FROM pose_landmarks", conn)

# 기본 정보 출력
print(f"총 레코드 수: {len(df)}")
print(f"총 컬럼 수: {len(df.columns)}")
print("\n라벨별 분포:")
print(df['label'].value_counts().sort_index())

# 사람별 분포 출력
if 'person_id' in df.columns:
    print("\n사람별 분포:")
    print(df['person_id'].value_counts().sort_index())
else:
    print("\nperson_id 컬럼이 없습니다. 기존 데이터입니다.")

# CSV 파일로 저장
csv_filename = os.path.join(csv_dir, 'pose_landmarks_data.csv')
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print(f"\nCSV 파일이 생성되었습니다: {csv_filename}")
print(f"파일 크기: {os.path.getsize(csv_filename) / 1024:.1f} KB")

# 라벨별로 분리된 CSV 파일도 생성
print("\n라벨별로 분리된 CSV 파일도 생성합니다...")

# 정면 데이터 (라벨 1)
front_df = df[df['label'] == 1]
if len(front_df) > 0:
    front_csv = os.path.join(csv_dir, 'pose_landmarks_front.csv')
    front_df.to_csv(front_csv, index=False, encoding='utf-8-sig')
    print(f"정면 데이터: {front_csv} ({len(front_df)}개 레코드)")

# 측면 데이터 (라벨 2)
side_df = df[df['label'] == 2]
if len(side_df) > 0:
    side_csv = os.path.join(csv_dir, 'pose_landmarks_side.csv')
    side_df.to_csv(side_csv, index=False, encoding='utf-8-sig')
    print(f"측면 데이터: {side_csv} ({len(side_df)}개 레코드)")

# 제외 데이터 (라벨 3)
exclude_df = df[df['label'] == 3]
if len(exclude_df) > 0:
    exclude_csv = os.path.join(csv_dir, 'pose_landmarks_exclude.csv')
    exclude_df.to_csv(exclude_csv, index=False, encoding='utf-8-sig')
    print(f"제외 데이터: {exclude_csv} ({len(exclude_df)}개 레코드)")

# 사람별로 분리된 CSV 파일 생성
if 'person_id' in df.columns:
    print("\n사람별로 분리된 CSV 파일도 생성합니다...")
    unique_persons = df['person_id'].unique()
    
    for person in unique_persons:
        person_df = df[df['person_id'] == person]
        if len(person_df) > 0:
            filename = os.path.join(csv_dir, f'pose_landmarks_{person}.csv')
            person_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"{person} 데이터: {filename} ({len(person_df)}개 레코드)")
            
            # 사람별 라벨 분포도 출력
            label_col = person_df['label']
            try:
                label_counts = pd.Series(label_col).value_counts().sort_index()
                print(f"  - 라벨 분포: {dict(label_counts)}")
            except Exception as e:
                print(f"  - 라벨 분포 계산 오류: {e}")

# 요약 정보를 별도 파일로 저장
summary_data = {
    '항목': ['총 레코드 수', '정면 데이터', '측면 데이터', '제외 데이터', '총 컬럼 수']
}
summary_values = [str(len(df)), str(len(front_df)), str(len(side_df)), str(len(exclude_df)), str(len(df.columns))]

# 사람별 요약 정보 추가
if 'person_id' in df.columns:
    unique_persons = df['person_id'].unique()
    for person in unique_persons:
        person_count = str(len(df[df['person_id'] == person]))
        summary_data['항목'].append(f'{person} 데이터')
        summary_values.append(person_count)

summary_data['값'] = summary_values
summary_df = pd.DataFrame(summary_data)
summary_csv = os.path.join(csv_dir, 'pose_landmarks_summary.csv')
summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
print(f"요약 정보: {summary_csv}")

conn.close()
print("\n=== CSV 파일 생성 완료 ===")
print("생성된 파일들:")
print(f"1. {csv_filename} - 전체 데이터")
print(f"2. {os.path.join(csv_dir, 'pose_landmarks_front.csv')} - 정면 데이터만")
print(f"3. {os.path.join(csv_dir, 'pose_landmarks_side.csv')} - 측면 데이터만")
print(f"4. {os.path.join(csv_dir, 'pose_landmarks_exclude.csv')} - 제외 데이터만")
if 'person_id' in df.columns:
    unique_persons = df['person_id'].unique()
    for i, person in enumerate(unique_persons, 5):
        print(f"{i}. {os.path.join(csv_dir, f'pose_landmarks_{person}.csv')} - {person} 데이터만")
    print(f"{len(unique_persons) + 5}. {summary_csv} - 요약 정보")
else:
    print(f"5. {summary_csv} - 요약 정보") 