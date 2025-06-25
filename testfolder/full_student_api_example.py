
from flask import Flask, request, jsonify
import sqlite3
import pandas as pd

app = Flask(__name__)

DB_PATH = 'full_student_project.db'

@app.route('/api/login', methods=['POST'])
def login():
    username = request.json.get('username')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return jsonify({"user_id": user[0], "username": user[1]})
    else:
        return jsonify({"error": "User not found"}), 404

@app.route('/api/posture', methods=['POST'])
def add_posture():
    data = request.json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO posture_logs (user_id, posture_score, captured_at) VALUES (?, ?, ?)",
                   (data['user_id'], data['score'], data['captured_at']))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

@app.route('/api/posture', methods=['GET'])
def get_posture():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM posture_logs", conn)
    conn.close()
    return df.to_json(orient='records', force_ascii=False)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM posture_logs", conn)
    df_grouped = df.groupby('captured_at')['posture_score'].mean().reset_index()
    conn.close()
    return df_grouped.to_json(orient='records', force_ascii=False)

@app.route('/api/goal', methods=['POST'])
def set_goal():
    data = request.json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO goals (user_id, daily_goal_minutes) VALUES (?, ?)",
                   (data['user_id'], data['daily_goal_minutes']))
    conn.commit()
    conn.close()
    return jsonify({"status": "goal set"})

@app.route('/api/badge', methods=['GET'])
def get_badges():
    user_id = request.args.get('user_id')
    conn = sqlite3.connect(DB_PATH)
    query = '''
    SELECT b.badge_name, b.description, ub.earned_at
    FROM user_badges ub
    JOIN badges b ON ub.badge_id = b.badge_id
    WHERE ub.user_id = ?
    '''
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df.to_json(orient='records', force_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)
