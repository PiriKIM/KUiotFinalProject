import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# ✅ 1) DB 연결 (파일 경로 정확!)
db_path = '/home/bangme/bangme/KUiotFinalProject/testfolder/my_posture.db'
conn = sqlite3.connect(db_path)

# ✅ 2) 테이블 확인 (디버그)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("📌 현재 DB 테이블:", cursor.fetchall())

# ✅ 3) Pandas로 데이터 읽기
df = pd.read_sql_query("SELECT * FROM posture_logs", conn)
print("\n📌 posture_logs 테이블 데이터:")
print(df)

# ✅ 4) 그룹 통계: 날짜별 평균 점수
df_grouped = df.groupby('captured_at')['posture_score'].mean().reset_index()
print("\n📌 날짜별 평균 점수:")
print(df_grouped)

# ✅ 5) 그래프
plt.plot(df_grouped['captured_at'], df_grouped['posture_score'], marker='o')
plt.title('날짜별 평균 자세 점수')
plt.xlabel('날짜')
plt.ylabel('평균 점수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ✅ 6) 연결 닫기
conn.close()
