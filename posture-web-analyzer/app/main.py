from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from app.analyzer.posture import PostureAnalyzer
from app.models.posture_result import PostureResult

app = FastAPI()
posture_analyzer = PostureAnalyzer()
mp_pose = mp.solutions.pose

@app.post("/analyze", response_model=PostureResult)
async def analyze_posture(file: UploadFile = File(...)):
    # 업로드된 이미지를 numpy array로 변환
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"error": "이미지 파일을 읽을 수 없습니다."})

    # MediaPipe Pose로 자세 추정
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return JSONResponse(status_code=200, content={"detected": False, "message": "사람이 감지되지 않았습니다."})

        # 자세 분석
        landmarks = results.pose_landmarks.landmark
        neck_result = posture_analyzer.analyze_turtle_neck_detailed(landmarks)
        spine_result = posture_analyzer.analyze_spine_curvature(landmarks)
        shoulder_result = posture_analyzer.analyze_shoulder_asymmetry(landmarks)
        pelvic_result = posture_analyzer.analyze_pelvic_tilt(landmarks)
        spine_twisting_result = posture_analyzer.analyze_spine_twisting(landmarks)

        # 결과를 JSON 형태로 반환 (Pydantic 모델 사용 권장)
        return PostureResult(
            detected=True,
            neck=neck_result,
            spine=spine_result,
            shoulder=shoulder_result,
            pelvic=pelvic_result,
            spine_twisting=spine_twisting_result
        )