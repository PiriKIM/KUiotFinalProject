#Flask 서버

from flask import Flask, render_template, request, jsonify
from utils.neck_analyzer import analyze_posture  # neck.py에서 함수 분리 필요
import cv2
import numpy as np
import base64
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from yj.Back_End.MediaPipe_test.neck import PostureAnalyzer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json['image']
    img_bytes = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = analyze_posture(img)  # 분석 함수 호출
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)