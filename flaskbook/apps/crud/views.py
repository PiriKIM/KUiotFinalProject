from flask import Blueprint, render_template, request, jsonify
import numpy as np
import cv2
from .neck import PostureAnalyzer
import mediapipe as mp

crud = Blueprint(
    'crud',
    __name__,
    template_folder='templates',
    static_folder='static'
)

analyzer = PostureAnalyzer()

@crud.route('/')
def index():
    return render_template('crud/index.html')

@crud.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            return jsonify({
                'neck': analyzer.analyze_turtle_neck_detailed(lm),
                'spine': analyzer.analyze_spine_curvature(lm),
                'shoulder': analyzer.analyze_shoulder_asymmetry(lm),
                'pelvic': analyzer.analyze_pelvic_tilt(lm),
                'twist': analyzer.analyze_spine_twisting(lm),
            })
        else:
            return jsonify({'error': 'No person detected'})
