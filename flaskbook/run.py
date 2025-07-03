from flask import redirect, url_for, session
import threading
import webbrowser
import time

# Flask 앱 생성
from apps.app import create_app
app = create_app()

# ESP32-CAM 영상 분석 함수
from apps.crud.utils.esp_module import run_pose_tracking

@app.route('/')
def root():
    # 로그인하지 않은 사용자는 로그인 페이지로 리다이렉트
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    # 로그인한 사용자는 메인 페이지로 리다이렉트
    return redirect(url_for('crud.index'))

@app.route('/start-stream')
def start_stream():
    thread = threading.Thread(target=run_pose_tracking)
    thread.daemon = True
    thread.start()
    return redirect(url_for('crud.index'))

def open_browser():
    time.sleep(1)
    webbrowser.open("http://localhost:5000")

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
