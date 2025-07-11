from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
import time

app = Flask(__name__)

# MediaPipe pose 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 기본 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 클라이언트에서 전송한 이미지 처리
@app.route('/upload', methods=['POST'])
def upload():
    data = request.json

    # base64 인코딩된 이미지 데이터를 ',' 기준으로 분리 후 디코딩
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)

    # 이미지 디코딩 (OpenCV)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # MediaPipe로 사람 인식 (RGB 변환 필수)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 사람 인식 여부 반환
    if results.pose_landmarks:
        return jsonify({'status': 'success', 'message': '사람 인식 성공'})
    else:
        return jsonify({'status': 'fail', 'message': '사람 인식 실패'})

if __name__ == '__main__':
    # HTTPS 실행: 로컬 인증서 경로 지정
    # cert.pem, key.pem은 mkcert 등으로 생성해야 함
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        ssl_context=('cert.pem', 'key.pem')  # HTTPS 인증서 적용
    )
