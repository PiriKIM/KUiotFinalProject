from flask import Flask, request
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
mp_pose = mp.solutions.pose

@app.route("/upload", methods=["POST"])
def upload():
    img_array = np.frombuffer(request.data, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            return "person"
        else:
            return "none"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
