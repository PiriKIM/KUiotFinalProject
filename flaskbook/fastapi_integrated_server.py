from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import aiohttp
import cv2
import numpy as np
import mediapipe as mp
import json
import sqlite3
import threading
import time
import subprocess
import logging
import asyncio
import re
import math
from datetime import datetime
from typing import Optional
import os
from pydantic import BaseModel

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ESP32-CAM 자세 분석 시스템", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="apps/crud/static"), name="static")
templates = Jinja2Templates(directory="apps/templates")

# ESP32 설정
ESP32_IP = "192.168.0.102"
ESP32_STREAM_PORT = 81
ESP32_API_PORT = 81  # ESP32-CAM은 81 포트에서 모든 서비스 제공
ESP32_STREAM_URL = f"http://{ESP32_IP}:{ESP32_STREAM_PORT}"
ESP32_API_URL = f"http://{ESP32_IP}:{ESP32_API_PORT}"

# ESP8266 Buzzer 설정
ESP8266_IP = "192.168.0.112"
ESP8266_PORT = 80
ESP8266_BUZZER_URL = f"http://{ESP8266_IP}:{ESP8266_PORT}"

# MediaPipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 데이터베이스 초기화
def init_db():
    conn = sqlite3.connect('local.sqlite')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posture_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            overall_score INTEGER,
            overall_grade TEXT,
            neck_angle REAL,
            neck_grade TEXT,
            spine_angle REAL,
            shoulder_angle REAL,
            pelvic_angle REAL,
            user_id INTEGER DEFAULT 1
        )
    ''')
    conn.commit()
    conn.close()

# ESP32 연결 확인
def check_esp32_connection():
    try:
        response = requests.get(f"{ESP32_API_URL}/status", timeout=2)
        return response.status_code == 200
    except:
        return False

# 연속 자세 분석 카운터
posture_counter = {
    'A': 0,
    'B': 0, 
    'C': 0,
    'no_pose': 0
}

# 전역 스트림 연결 관리
class StreamManager:
    def __init__(self):
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.stream_active = False
        self.last_frame_time = 0
        
    def update_frame(self, frame):
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None
            self.last_frame_time = time.time()
    
    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None and time.time() - self.last_frame_time < 5:  # 5초 이내 프레임만 유효
                return self.current_frame.copy()
            return None
    
    def is_stream_active(self):
        return self.stream_active and (time.time() - self.last_frame_time < 10)  # 10초 이내 활동 있으면 활성

# 전역 스트림 매니저 인스턴스
stream_manager = StreamManager()

# ESP8266 Buzzer Control 함수들
def check_esp8266_connection():
    """ESP8266 연결 상태 확인"""
    try:
        response = requests.get(f"{ESP8266_BUZZER_URL}/status", timeout=2)
        return response.status_code == 200
    except:
        return False

def trigger_buzzer(duration_ms=1000, volume=50):
    """ESP8266 buzzer 트리거 (동기)"""
    try:
        data = {
            "duration": duration_ms,
            "volume": volume
        }
        response = requests.post(f"{ESP8266_BUZZER_URL}/buzzer", json=data, timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Buzzer trigger failed: {e}")
        return False

async def trigger_buzzer_async(duration_ms=1000, volume=50):
    """ESP8266 buzzer 트리거 (비동기) - aiohttp 사용"""
    try:
        data = {
            "duration": duration_ms,
            "volume": volume
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{ESP8266_BUZZER_URL}/buzzer", json=data, timeout=aiohttp.ClientTimeout(total=2)) as response:
                if response.status == 200:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"[{current_time}] 🔊 비동기 부저 트리거 성공: {duration_ms}ms, 볼륨 {volume}%")
                    return True
                else:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.error(f"[{current_time}] ❌ 비동기 부저 트리거 실패: {response.status}")
                    return False
    except Exception as e:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"[{current_time}] ❌ 비동기 부저 트리거 오류: {e}")
        return False

def set_buzzer_volume(volume):
    """ESP8266 buzzer 볼륨 설정"""
    try:
        data = {"volume": volume}
        response = requests.post(f"{ESP8266_BUZZER_URL}/volume", json=data, timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Buzzer volume setting failed: {e}")
        return False

def get_buzzer_status():
    """ESP8266 buzzer 상태 확인"""
    try:
        response = requests.get(f"{ESP8266_BUZZER_URL}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Buzzer status check failed: {e}")
        return None

# Flask의 PostureAnalyzer 클래스 가져오기
class PostureAnalyzer:
    def __init__(self):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        """세 점으로 각도 계산"""
        a_pt = np.array([a.x, a.y]) if not isinstance(a, np.ndarray) else a
        b_pt = np.array([b.x, b.y]) if not isinstance(b, np.ndarray) else b
        c_pt = np.array([c.x, c.y]) if not isinstance(c, np.ndarray) else c

        ba = a_pt - b_pt
        bc = c_pt - b_pt

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_neck_angle(self, landmarks):
        """목 각도 계산"""
        mp = self.mp_pose
        try:
            if landmarks[mp.PoseLandmark.LEFT_EAR].visibility > 0.5 and landmarks[mp.PoseLandmark.LEFT_SHOULDER].visibility > 0.5:
                ear = landmarks[mp.PoseLandmark.LEFT_EAR]
                shoulder = landmarks[mp.PoseLandmark.LEFT_SHOULDER]
            elif landmarks[mp.PoseLandmark.RIGHT_EAR].visibility > 0.5 and landmarks[mp.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5:
                ear = landmarks[mp.PoseLandmark.RIGHT_EAR]
                shoulder = landmarks[mp.PoseLandmark.RIGHT_SHOULDER]
            else:
                left_ear = landmarks[mp.PoseLandmark.LEFT_EAR]
                right_ear = landmarks[mp.PoseLandmark.RIGHT_EAR]
                left_shoulder = landmarks[mp.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp.PoseLandmark.RIGHT_SHOULDER]
                ear = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])
                shoulder = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])

            if hasattr(ear, 'x'):
                vertical = np.array([shoulder.x, ear.y])
            else:
                vertical = np.array([shoulder[0], ear[1]])
            return self.calculate_angle(ear, shoulder, vertical)
        except:
            return None

    def grade_neck_posture(self, neck_angle):
        if neck_angle <= 5:
            return 'A', "완벽한 자세"
        elif neck_angle <= 10:
            return 'B', "양호한 자세"
        elif neck_angle <= 15:
            return 'C', "보통 자세"
        else:
            return 'D', "나쁜 자세"

    def analyze_turtle_neck_detailed(self, landmarks):
        """거북목 상세 분석"""
        mp = self.mp_pose
        neck_angle = self.calculate_neck_angle(landmarks)
        grade, desc = self.grade_neck_posture(neck_angle)
        neck_top = (
            (landmarks[mp.PoseLandmark.LEFT_EAR].x + landmarks[mp.PoseLandmark.RIGHT_EAR].x) / 2,
            (landmarks[mp.PoseLandmark.LEFT_EAR].y + landmarks[mp.PoseLandmark.RIGHT_EAR].y) / 2
        )
        shoulder_center = (
            (landmarks[mp.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp.PoseLandmark.RIGHT_SHOULDER].x) / 2,
            (landmarks[mp.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp.PoseLandmark.RIGHT_SHOULDER].y) / 2
        )
        vertical_deviation = abs(neck_top[0] - shoulder_center[0])
        return {
            'neck_angle': neck_angle,
            'grade': grade,
            'grade_description': desc,
            'vertical_deviation': vertical_deviation,
            'neck_top': neck_top,
            'shoulder_center': shoulder_center
        }

    def analyze_spine_curvature(self, landmarks):
        """척추 굴곡 분석"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 어깨와 엉덩이 중심점
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # 척추 각도 계산
        spine_angle = math.degrees(math.atan2(hip_center_x - shoulder_center_x, 
                                            hip_center_y - shoulder_center_y))
        
        # 척추 굴곡 판정
        is_hunched = abs(spine_angle) > 12
        
        return {
            'is_hunched': is_hunched,
            'spine_angle': spine_angle,
            'shoulder_center': (shoulder_center_x, shoulder_center_y),
            'hip_center': (hip_center_x, hip_center_y)
        }

    def analyze_shoulder_asymmetry(self, landmarks):
        """어깨 비대칭 분석"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # 어깨 높이 차이 계산
        shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # 어깨 비대칭 판정
        is_asymmetric = shoulder_height_diff > 0.02
        
        return {
            'is_asymmetric': is_asymmetric,
            'height_difference': shoulder_height_diff,
            'left_shoulder': (left_shoulder.x, left_shoulder.y),
            'right_shoulder': (right_shoulder.x, right_shoulder.y)
        }

    def analyze_pelvic_tilt(self, landmarks):
        """골반 기울기 분석"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 골반 높이 차이 계산
        hip_height_diff = abs(left_hip.y - right_hip.y)
        
        # 골반 기울어짐 판정
        is_tilted = hip_height_diff > 0.015
        
        return {
            'is_tilted': is_tilted,
            'height_difference': hip_height_diff,
            'left_hip': (left_hip.x, left_hip.y),
            'right_hip': (right_hip.x, right_hip.y)
        }

# PostureAnalyzer 인스턴스 생성
posture_analyzer = PostureAnalyzer()

def analyze_posture(landmarks):
    """완전한 자세 분석 (Flask 버전과 동일)"""
    if not landmarks:
        return {
            'overall_score': 50,
            'overall_grade': 'C',
            'neck': {'angle': None, 'grade': 'C', 'description': '자세 감지 안됨'},
            'spine': {'is_hunched': False},
            'shoulder': {'is_asymmetric': False},
            'pelvic': {'is_tilted': False}
        }
    
    # 각 부위별 분석
    neck_result = posture_analyzer.analyze_turtle_neck_detailed(landmarks)
    spine_result = posture_analyzer.analyze_spine_curvature(landmarks)
    shoulder_result = posture_analyzer.analyze_shoulder_asymmetry(landmarks)
    pelvic_result = posture_analyzer.analyze_pelvic_tilt(landmarks)
    
    # 종합 점수 계산 (Flask와 동일한 로직)
    score = 100
    
    # 목 자세 점수 (30점)
    if neck_result['grade'] == 'A':
        score -= 0
    elif neck_result['grade'] == 'B':
        score -= 10
    elif neck_result['grade'] == 'C':
        score -= 20
    else:  # D
        score -= 30
    
    # 척추 곡률 점수 (25점)
    if spine_result['is_hunched']:
        score -= 25
    
    # 어깨 비대칭 점수 (20점)
    if shoulder_result['is_asymmetric']:
        score -= 20
    
    # 골반 기울기 점수 (15점)
    if pelvic_result['is_tilted']:
        score -= 15
    
    score = max(0, score)
    
    # 종합 등급
    if score >= 90:
        overall_grade = 'A'
    elif score >= 70:
        overall_grade = 'B'
    else:
        overall_grade = 'C'
    
    return {
        'overall_score': score,
        'overall_grade': overall_grade,
        'neck': neck_result,
        'spine': spine_result,
        'shoulder': shoulder_result,
        'pelvic': pelvic_result
    }

# 데이터베이스 저장
def save_posture_data(analysis_result):
    try:
        conn = sqlite3.connect('local.sqlite')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO posture_data 
            (overall_score, overall_grade, neck_angle, neck_grade)
            VALUES (?, ?, ?, ?)
        ''', (
            analysis_result['overall_score'],
            analysis_result['overall_grade'],
            analysis_result['neck_angle'],
            analysis_result['neck_grade']
        ))
        conn.commit()
        conn.close()
        logger.info("데이터베이스 저장 성공")
        return True
    except Exception as e:
        logger.error(f"데이터베이스 저장 실패: {e}")
        return False

# 라우트 정의
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        return templates.TemplateResponse("crud/index.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"""
        <html>
        <head><title>FastAPI 통합 서버</title></head>
        <body>
            <h1>🚀 FastAPI 통합 서버가 실행 중입니다!</h1>
            <p>📡 ESP32-CAM IP: {ESP32_IP}</p>
            <p>🌐 서버 주소: http://localhost:8000</p>
            <p>✅ ESP32 연결 상태: {'연결됨' if check_esp32_connection() else '연결되지 않음'}</p>
            <hr>
            <p><a href="/crud/">자세 분석 페이지로 이동</a></p>
            <p><a href="/crud/buzzer-test">부저 테스트</a></p>
            <p><strong>오류 정보:</strong> {str(e)}</p>
        </body>
        </html>
        """)

@app.get("/auth/login", response_class=HTMLResponse, name="auth.login")
async def login_page(request: Request):
    return templates.TemplateResponse("crud/login.html", {"request": request})

@app.post("/auth/login")
async def login(request: Request):
    return JSONResponse({"success": True})

@app.get("/auth/register", response_class=HTMLResponse, name="auth.register")
async def register_page(request: Request):
    return templates.TemplateResponse("crud/register.html", {"request": request})

@app.get("/auth/logout", response_class=HTMLResponse, name="auth.logout")
async def logout_page(request: Request):
    # 실제 로그아웃 처리는 추후 구현
    return templates.TemplateResponse("crud/login.html", {"request": request})

@app.get("/crud/", response_class=HTMLResponse, name="crud.index")
async def crud_index(request: Request):
    return templates.TemplateResponse("crud/index.html", {"request": request, "user": None})

@app.get("/esp32-stream")
async def esp32_stream():
    def generate():
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logger.info(f"ESP32 스트림 연결 시도 {retry_count + 1}/{max_retries}")
                response = requests.get(f"{ESP32_STREAM_URL}/stream", stream=True, timeout=10)
                
                if response.status_code == 200:
                    logger.info("ESP32 스트림 연결 성공")
                    stream_manager.stream_active = True
                    
                    content = b""
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            content += chunk
                            yield chunk
                            
                            # 프레임 추출 및 저장 (분석용)
                            start_pos = content.find(b'\xff\xd8')
                            if start_pos != -1:
                                end_pos = content.find(b'\xff\xd9', start_pos)
                                if end_pos != -1:
                                    jpeg_data = content[start_pos:end_pos + 2]
                                    image_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        stream_manager.update_frame(frame)
                                    content = content[end_pos + 2:]  # 처리된 프레임 제거
                    
                    break  # 성공하면 루프 종료
                else:
                    logger.error(f"ESP32 스트림 연결 실패: {response.status_code}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2)  # 2초 대기 후 재시도
                        
            except Exception as e:
                logger.error(f"ESP32 스트림 오류 (시도 {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)  # 2초 대기 후 재시도
        
        if retry_count >= max_retries:
            logger.error("ESP32 스트림 연결 최대 재시도 횟수 초과")
            stream_manager.stream_active = False
            # 에러 이미지나 메시지 반환
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 0\r\n\r\n"
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/esp32-stream-analyze")
async def esp32_stream_with_analysis():
    """실시간 스트림 (프레임 저장 포함)"""
    def generate():
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logger.info(f"ESP32 분석 스트림 연결 시도 {retry_count + 1}/{max_retries}")
                response = requests.get(f"{ESP32_STREAM_URL}/stream", stream=True, timeout=10)
                
                if response.status_code == 200:
                    logger.info("ESP32 분석 스트림 연결 성공")
                    stream_manager.stream_active = True
                    
                    content = b""
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            content += chunk
                            yield chunk
                            # MJPEG 스트림에서 JPEG 프레임 추출 및 저장
                            while True:
                                start = content.find(b'\xff\xd8')
                                end = content.find(b'\xff\xd9', start)
                                if start != -1 and end != -1:
                                    jpeg = content[start:end+2]
                                    image_array = np.frombuffer(jpeg, dtype=np.uint8)
                                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        stream_manager.update_frame(frame)
                                    content = content[end+2:]
                                else:
                                    break
                    
                    break  # 성공하면 루프 종료
                else:
                    logger.error(f"ESP32 분석 스트림 연결 실패: {response.status_code}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2)  # 2초 대기 후 재시도
                        
            except Exception as e:
                logger.error(f"ESP32 분석 스트림 오류 (시도 {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)  # 2초 대기 후 재시도
        
        if retry_count >= max_retries:
            logger.error("ESP32 분석 스트림 연결 최대 재시도 횟수 초과")
            stream_manager.stream_active = False
            # 에러 이미지나 메시지 반환
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 0\r\n\r\n"
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# 요청 모델 추가
class AnalyzeRequest(BaseModel):
    capture_frame: bool = True

@app.post("/crud/analyze")
async def analyze_posture_endpoint(request: Request):
    """실시간 분석 (저장된 프레임 사용)"""
    try:
        # 스트림 매니저에서 최신 프레임 가져오기
        if not stream_manager.is_stream_active():
            return JSONResponse({
                "error": "ESP32-CAM 스트림이 활성화되지 않았습니다",
                "state": "error"
            })
        
        image = stream_manager.get_frame()
        if image is None:
            return JSONResponse({
                "error": "유효한 프레임을 찾을 수 없습니다",
                "state": "error"
            })
        
        # MediaPipe 분석
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # 자세 분석 (자세 감지 안됨도 C등급으로 처리)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            analysis_result = analyze_posture(landmarks)
        else:
            # 자세 감지 안됨 → C등급으로 처리
            analysis_result = analyze_posture(None)
        
        if not analysis_result:
            return JSONResponse({
                "error": "자세 분석에 실패했습니다",
                "state": "analysis_failed"
            })
        
        # 연속 자세 카운터 업데이트
        grade = analysis_result['overall_grade']
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for g in ['A', 'B', 'C']:
            if g == grade:
                posture_counter[g] += 1
            else:
                posture_counter[g] = 0
        
        # 부저 트리거 로직
        buzzer_triggered = False
        
        # B등급 30번 연속 → 부저 1번
        if posture_counter['B'] == 30:
            buzzer_status = get_buzzer_status()
            volume = 50  # 기본값
            if buzzer_status and 'volume' in buzzer_status:
                volume = buzzer_status['volume']
            
            logger.info(f"[{current_time}] 🚨 B등급 30번 연속! 부저 1번 울림 (볼륨: {volume}%)")
            trigger_buzzer(duration_ms=1000, volume=volume)
            posture_counter['B'] = 0  # 카운터 리셋
            buzzer_triggered = True
        
        # C등급 30번 연속 → 부저 2번
        elif posture_counter['C'] == 30:
            buzzer_status = get_buzzer_status()
            volume = 50  # 기본값
            if buzzer_status and 'volume' in buzzer_status:
                volume = buzzer_status['volume']
            
            logger.info(f"[{current_time}] 🚨 C등급 30번 연속! 부저 2번 울림 (볼륨: {volume}%)")
            # 부저 2번 연속 울림
            trigger_buzzer(duration_ms=1000, volume=volume)
            time.sleep(0.1)  # 0.1초 간격
            trigger_buzzer(duration_ms=1000, volume=volume)
            posture_counter['C'] = 0  # 카운터 리셋
            buzzer_triggered = True
        
        # 데이터베이스에 저장
        save_posture_data(analysis_result)
        
        # 응답 데이터 구성
        response_data = {
            "overall_score": analysis_result['overall_score'],
            "overall_grade": analysis_result['overall_grade'],
            "neck": analysis_result.get('neck', {}),
            "spine": analysis_result.get('spine', {}),
            "shoulder": analysis_result.get('shoulder', {}),
            "pelvic": analysis_result.get('pelvic', {}),
            "buzzer_triggered": buzzer_triggered,
            "state": "analyzed",
            "state_message": f"분석 완료: {analysis_result['overall_score']}점 ({grade}등급)",
            "timestamp": current_time
        }
        
        logger.info(f"[{current_time}] 분석 완료: {analysis_result['overall_score']}점 ({grade}등급) - 연속: A({posture_counter['A']}) B({posture_counter['B']}) C({posture_counter['C']})")
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logger.error(f"분석 엔드포인트 오류: {e}")
        return JSONResponse({
            "error": f"분석 중 오류가 발생했습니다: {str(e)}",
            "state": "error"
        })

@app.get("/crud/statistics", response_class=HTMLResponse, name="crud.statistics")
async def statistics_page(request: Request):
    return templates.TemplateResponse("crud/statistics.html", {"request": request, "user": None, "total_records": 0, "avg_score": 0, "grade_counts": {"A":0, "B":0, "C":0, "D":0}})

@app.get("/crud/history", response_class=HTMLResponse, name="crud.history")
async def history_page(request: Request):
    return templates.TemplateResponse("crud/history.html", {"request": request, "user": None, "records": {"items": [], "pages": 1, "page": 1, "has_prev": False, "has_next": False, "prev_num": 1, "next_num": 1}})

@app.get("/crud/profile", response_class=HTMLResponse, name="auth.profile")
async def profile_page(request: Request):
    return templates.TemplateResponse("crud/profile.html", {"request": request, "user": None, "total_analyses": 0, "avg_score": 0, "grade_counts": {"A":0, "B":0, "C":0, "D":0}, "recent_records": []})

@app.get("/crud/buzzer-test", response_class=HTMLResponse, name="crud.buzzer_test")
async def buzzer_test_page(request: Request):
    return templates.TemplateResponse("crud/buzzer_test.html", {"request": request})

# ESP8266 Buzzer Control API 엔드포인트들
class VolumeRequest(BaseModel):
    volume: int

class TriggerRequest(BaseModel):
    duration: int = 1000
    volume: int = 50

@app.post("/api/buzzer/trigger")
async def api_trigger_buzzer(req: TriggerRequest):
    """API를 통해 buzzer 트리거"""
    success = trigger_buzzer(duration_ms=req.duration, volume=req.volume)
    return JSONResponse({
        "success": success,
        "message": "Buzzer triggered" if success else "Failed to trigger buzzer"
    })

@app.post("/api/buzzer/volume")
async def api_set_buzzer_volume(req: VolumeRequest):
    volume = req.volume
    if not 0 <= volume <= 100:
        raise HTTPException(status_code=400, detail="Volume must be between 0 and 100")
    
    success = set_buzzer_volume(volume)
    return JSONResponse({
        "success": success,
        "message": f"Volume set to {volume}" if success else "Failed to set volume"
    })

@app.get("/api/buzzer/status")
async def api_get_buzzer_status():
    """API를 통해 buzzer 상태 확인"""
    status = get_buzzer_status()
    connection_status = check_esp8266_connection()
    
    return JSONResponse({
        "connected": connection_status,
        "status": status,
        "esp8266_ip": ESP8266_IP
    })

@app.get("/api/buzzer/test")
async def api_test_buzzer():
    """API를 통해 buzzer 테스트 (짧은 비프음) - 현재 설정된 볼륨 사용"""
    # ESP8266에서 현재 볼륨 상태 가져오기
    status = get_buzzer_status()
    current_volume = 50  # 기본값
    
    if status and 'volume' in status:
        current_volume = status['volume']
    
    success = trigger_buzzer(duration_ms=500, volume=current_volume)
    return JSONResponse({
        "success": success,
        "message": f"Test beep sent with volume {current_volume}%" if success else "Failed to send test beep"
    })

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("🚀 FastAPI 통합 서버 시작")
    logger.info(f"📡 ESP32-CAM IP 주소: {ESP32_IP}")
    logger.info("🌐 브라우저에서 http://localhost:8000 접속")

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_integrated_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 