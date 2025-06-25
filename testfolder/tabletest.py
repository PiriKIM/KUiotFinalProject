import sqlite3

# ✅ DB 연결 (없으면 새로 생성됨)
conn = sqlite3.connect('my_posture.db')
cursor = conn.cursor()

# ✅ 테이블 생성
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS posture_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    posture_score INTEGER,
    captured_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS correction_history (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    correction_note TEXT,
    corrected_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
)
""")

# ✅ 샘플 데이터 추가 (원하면)
cursor.execute("INSERT INTO users (username) VALUES (?)", ('park_eojin',))
user_id = cursor.lastrowid

cursor.execute("INSERT INTO posture_logs (user_id, posture_score, captured_at) VALUES (?, ?, ?)",
               (user_id, 80, "2024-06-16"))

cursor.execute("INSERT INTO correction_history (user_id, correction_note, corrected_at) VALUES (?, ?, ?)",
               (user_id, "어깨를 펴세요", "2024-06-16"))

conn.commit()

# ✅ 현재 테이블 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("📌 현재 DB 테이블:", cursor.fetchall())

conn.close()
