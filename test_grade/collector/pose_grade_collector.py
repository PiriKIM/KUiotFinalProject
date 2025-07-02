import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import json
import time
from datetime import datetime
import os
import sys
import threading
import queue
from PIL import Image, ImageDraw, ImageFont
import pickle

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.posture_analyzer import PostureAnalyzer
from utils.text_utils import put_korean_text

# 전역 폰트 변수
_global_font = None
_font_loaded = False

def get_korean_font(font_size=20):
    """한글 폰트를 한 번만 로드하여 재사용"""
    global _global_font, _font_loaded
    
    if _font_loaded and _global_font:
        return _global_font
    
    # 한글 폰트 로드 (여러 폰트 경로 시도)
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "C:/Windows/Fonts/malgun.ttf",      # Windows
        "/home/woo/kuBig2025/opencv/data/NanumPenScript-Regular.ttf"  # 사용자 지정 경로
    ]
    
    for font_path in font_paths:
        try:
            _global_font = ImageFont.truetype(font_path, font_size)
            _font_loaded = True
            print(f"한글 폰트 로드 성공: {font_path}")
            break
        except Exception as e:
            continue
    
    if not _font_loaded:
        # 폰트 로드 실패 시 기본 폰트 사용
        _global_font = ImageFont.load_default()
        print("기본 폰트 사용")
        _font_loaded = True
    
    return _global_font

def put_korean_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """PIL을 사용한 한글 텍스트 렌더링 함수"""
    try:
        # PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 폰트 가져오기
        font = get_korean_font(font_size)
        
        # 색상을 RGB로 변환 (PIL은 RGB 사용)
        color_rgb = (color[2], color[1], color[0])  # BGR to RGB
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=color_rgb)
        
        # OpenCV 이미지로 변환
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        print(f"PIL 한글 텍스트 렌더링 오류: {e}")
        # 오류 시 기본 OpenCV 텍스트 사용
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

class PoseGradeCollector:
    def __init__(self, db_path="pose_grade_data.db"):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.analyzer = PostureAnalyzer()
        self.db_path = db_path
        self.init_database()
        
        # 키보드 입력을 위한 변수들
        self.current_view = "1"  # 기본값: 정면
        self.auto_grade_mode = False  # 자동 등급 판별 모드
        self.key_queue = queue.Queue()
        self.running = True
        
        # 프레임 카운터 추가
        self.frame_count = 0
        
        # 데이터 저장을 위한 큐와 스레드
        self.save_queue = queue.Queue()
        self.save_thread = None
        self.save_thread_running = True
        
        # 자동 등급 판별 모델 로드
        self.grade_model = None
        self.feature_names = None
        self.load_grade_model()
        
    def init_database(self):
        """데이터베이스 초기화 - auto_grade 사용"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 기존 테이블 삭제 (스키마 변경을 위해)
        cursor.execute("DROP TABLE IF EXISTS pose_grade_data")
        
        # 33개 랜드마크의 x, y 좌표를 개별 컬럼으로 생성
        landmark_columns = []
        for i in range(33):
            landmark_columns.extend([f'landmark_{i}_x REAL', f'landmark_{i}_y REAL'])
        
        landmark_columns_str = ', '.join(landmark_columns)
        
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS pose_grade_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                participant_id TEXT,
                view_angle TEXT,
                auto_grade TEXT,
                neck_angle REAL,
                spine_angle REAL,
                shoulder_asymmetry REAL,
                pelvic_tilt REAL,
                total_score REAL,
                analysis_results TEXT,
                {landmark_columns_str}
            )
        ''')
        
        conn.commit()
        conn.close()
        print("데이터베이스 초기화 완료 - auto_grade 사용")
        
    def start_save_thread(self):
        """데이터 저장 스레드 시작"""
        self.save_thread = threading.Thread(target=self.save_worker)
        self.save_thread.daemon = True
        self.save_thread.start()
        print("데이터 저장 스레드 시작됨")
        
    def stop_save_thread(self):
        """데이터 저장 스레드 종료"""
        self.save_thread_running = False
        if self.save_thread:
            self.save_thread.join(timeout=2)
        print("데이터 저장 스레드 종료됨")
        
    def save_worker(self):
        """데이터 저장 워커 스레드"""
        while self.save_thread_running:
            try:
                # 큐에서 데이터 가져오기 (1초 타임아웃)
                data = self.save_queue.get(timeout=1)
                if data is None:  # 종료 신호
                    break
                    
                # 데이터베이스에 저장
                self._save_to_database(data)
                self.save_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"데이터 저장 오류: {e}")
                
    def _save_to_database(self, data):
        """실제 데이터베이스 저장 작업"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(data['query'], data['values'])
            conn.commit()
        except Exception as e:
            print(f"데이터베이스 저장 실패: {e}")
        finally:
            conn.close()
        
    def keyboard_input_handler(self):
        """키보드 입력 처리 스레드"""
        print("\n=== 키보드 입력 안내 ===")
        print("시점 변경:")
        print("  1: 정면")
        print("  2: 측면")
        print("  3: 정면도 측면도 아닌 상태")
        print("자동 등급 판별:")
        print("  auto: 자동 등급 판별 모드 토글")
        print("종료: q")
        print("======================\n")
        
        while self.running:
            try:
                key = input().lower().strip()
                if key in ['1', '2', '3', 'auto', 'q']:
                    self.key_queue.put(key)
                    if key == 'q':
                        self.running = False
                        break
            except KeyboardInterrupt:
                self.running = False
                break
        
    def process_key_input(self):
        """키보드 입력 처리 - auto_grade 모드로 변경"""
        try:
            if not self.key_queue.empty():
                key = self.key_queue.get_nowait()
                
                # 시점 변경 (1, 2, 3)
                if key in ['1', '2', '3']:
                    self.current_view = key
                    view_names = {'1': '정면', '2': '측면', '3': '기타 상태'}
                    print(f"시점 변경: {view_names.get(key, '알 수 없음')} ({key})")
                
                # 자동 등급 모드 토글
                elif key.lower() == 'auto':
                    self.auto_grade_mode = not self.auto_grade_mode
                    mode_status = "활성화" if self.auto_grade_mode else "비활성화"
                    print(f"자동 등급 판별 모드: {mode_status}")
                    
                    if self.auto_grade_mode and self.grade_model is None:
                        print("경고: 등급 판별 모델이 로드되지 않았습니다.")
                        print("모델을 훈련한 후 다시 시도하세요.")
                        self.auto_grade_mode = False
                
                # 종료
                elif key.lower() == 'q':
                    self.running = False
                    print("데이터 수집을 종료합니다.")
                    
        except queue.Empty:
            pass
        
    def collect_data_realtime(self, participant_id, duration=None):
        """실시간 자세 데이터 수집 (키보드 입력 기반) - 멀티스레딩 적용"""
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0
        
        print(f"실시간 자세 데이터 수집 시작...")
        print(f"참가자 ID: {participant_id}")
        print("키보드로 시점과 자세 등급을 실시간 변경하세요.")
        print("데이터 수집과 저장이 동시에 실행됩니다.")
        
        # 데이터 저장 스레드 시작
        self.start_save_thread()
        
        # 키보드 입력 스레드 시작
        keyboard_thread = threading.Thread(target=self.keyboard_input_handler)
        keyboard_thread.daemon = True
        keyboard_thread.start()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # 키보드 입력 처리
            self.process_key_input()
                
            # BGR to RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 10프레임마다 분석 수행
                if frame_count % 10 == 0:
                    # 자세 분석 수행
                    analysis = self.analyze_posture(landmarks)
                    
                    # 자동 등급 판별
                    auto_grade = None
                    total_score = None
                    if self.auto_grade_mode:
                        auto_grade, total_score = self.predict_grade(analysis)
                    
                    # auto_grade가 None인 경우 기본값 설정
                    if auto_grade is None:
                        auto_grade = 'c'  # 기본값: C등급
                        total_score = 70  # 기본 점수
                    
                    # 비동기 데이터 저장 (메인 스레드 블로킹 없음)
                    self.save_data_async(participant_id, landmarks, analysis, auto_grade, total_score)
                    
                    # 콘솔에 분석 결과 출력
                    self.print_analysis_to_console(analysis, auto_grade, total_score, frame_count)
                
                # 화면에 정보 표시 (모든 프레임)
                self.draw_info_realtime(frame, frame_count)
                frame_count += 1
            
            cv2.imshow('Real-time Pose Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # 정리 작업
        cap.release()
        cv2.destroyAllWindows()
        
        # 저장 스레드 종료
        self.stop_save_thread()
        
        # 남은 데이터 저장 완료 대기
        self.save_queue.join()
        
        print(f"데이터 수집 완료! 총 {frame_count}개 프레임 수집")
        
    def analyze_posture(self, landmarks):
        """자세 분석 수행"""
        analysis = {}
        
        # 목 자세 분석
        neck_analysis = self.analyzer.analyze_turtle_neck_detailed(landmarks)
        analysis['neck'] = neck_analysis
        
        # 척추 굴곡 분석
        spine_analysis = self.analyzer.analyze_spine_curvature(landmarks)
        analysis['spine'] = spine_analysis
        
        # 어깨 비대칭 분석
        shoulder_analysis = self.analyzer.analyze_shoulder_asymmetry(landmarks)
        analysis['shoulder'] = shoulder_analysis
        
        # 골반 기울기 분석
        pelvic_analysis = self.analyzer.analyze_pelvic_tilt(landmarks)
        analysis['pelvic'] = pelvic_analysis
        
        return analysis
        
    def save_data_async(self, participant_id, landmarks, analysis, auto_grade=None, total_score=None):
        """비동기 데이터 저장 - 큐에 저장 작업 추가"""
        # 33개 랜드마크를 66개 개별 컬럼으로 변환 (visibility 필터링 적용)
        landmark_values = []
        for i, landmark in enumerate(landmarks):
            # visibility < 0.5인 경우 -1로 저장
            if hasattr(landmark, 'visibility') and landmark.visibility < 0.5:
                landmark_values.extend([-1, -1])  # x, y 모두 -1
            else:
                landmark_values.extend([landmark.x, landmark.y])
        
        # 66개 랜드마크 컬럼명 생성
        landmark_columns = []
        for i in range(33):
            landmark_columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
        
        # SQL 쿼리 생성
        columns = ['timestamp', 'participant_id', 'view_angle', 'auto_grade',
                  'neck_angle', 'spine_angle', 'shoulder_asymmetry', 'pelvic_tilt', 
                  'total_score', 'analysis_results'] + landmark_columns
        
        placeholders = ['?'] * len(columns)
        
        values = [
            datetime.now().isoformat(),
            participant_id,
            self.current_view,
            auto_grade,
            analysis['neck']['neck_angle'],
            analysis['spine']['spine_angle'],
            analysis['shoulder']['height_difference'],
            analysis['pelvic']['height_difference'],
            total_score,
            json.dumps(analysis)
        ] + landmark_values
        
        # 저장 작업을 큐에 추가 (비동기)
        save_data = {
            'query': f'''
                INSERT INTO pose_grade_data 
                ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            ''',
            'values': values
        }
        
        try:
            self.save_queue.put_nowait(save_data)
        except queue.Full:
            print("저장 큐가 가득 찼습니다. 일부 데이터가 손실될 수 있습니다.")
        
    def save_data_realtime(self, participant_id, landmarks, analysis, auto_grade=None, total_score=None):
        """실시간 데이터 저장 - 66개 개별 컬럼으로 변경 (동기 버전 - 호환성용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 33개 랜드마크를 66개 개별 컬럼으로 변환 (visibility 필터링 적용)
        landmark_values = []
        for i, landmark in enumerate(landmarks):
            # visibility < 0.5인 경우 -1로 저장
            if hasattr(landmark, 'visibility') and landmark.visibility < 0.5:
                landmark_values.extend([-1, -1])  # x, y 모두 -1
            else:
                landmark_values.extend([landmark.x, landmark.y])
        
        # 66개 랜드마크 컬럼명 생성
        landmark_columns = []
        for i in range(33):
            landmark_columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
        
        # SQL 쿼리 생성
        columns = ['timestamp', 'participant_id', 'view_angle', 'auto_grade',
                  'neck_angle', 'spine_angle', 'shoulder_asymmetry', 'pelvic_tilt', 
                  'total_score', 'analysis_results'] + landmark_columns
        
        placeholders = ['?'] * len(columns)
        
        values = [
            datetime.now().isoformat(),
            participant_id,
            self.current_view,
            auto_grade,
            analysis['neck']['neck_angle'],
            analysis['spine']['spine_angle'],
            analysis['shoulder']['height_difference'],
            analysis['pelvic']['height_difference'],
            total_score,
            json.dumps(analysis)
        ] + landmark_values
        
        cursor.execute(f'''
            INSERT INTO pose_grade_data 
            ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        ''', values)
        
        conn.commit()
        conn.close()
        
    def print_analysis_to_console(self, analysis, auto_grade=None, total_score=None, frame_count=None):
        """콘솔에 분석 결과 출력"""
        print("\n" + "="*50)
        print(f"[프레임 {frame_count}] 자세 분석 결과")
        print("="*50)
        
        # 시점 정보
        view_names = {'1': '정면', '2': '측면', '3': '기타 상태'}
        view_name = view_names.get(self.current_view, '알 수 없음')
        print(f"시점: {view_name} ({self.current_view})")
        
        # 등급 정보
        pose_names = {'a': 'A등급 (완벽)', 'b': 'B등급 (양호)', 'c': 'C등급 (보통)', 
                     'd': 'D등급 (나쁨)', 'e': '특수 자세'}
        
        if self.auto_grade_mode and auto_grade:
            print(f"자세 등급: {pose_names.get(auto_grade, '알 수 없음')} (자동 판별)")
            if total_score is not None:
                print(f"종합 점수: {total_score:.1f}/100")
        else:
            print(f"자세 등급: {pose_names.get(auto_grade, '알 수 없음')} (기본값)")
            if total_score is not None:
                print(f"종합 점수: {total_score:.1f}/100")
        
        # 자세 분석 상세 정보
        print(f"\n[상세 분석]")
        print(f"목 각도: {analysis['neck']['neck_angle']:.1f}°")
        print(f"척추 각도: {analysis['spine']['spine_angle']:.1f}°")
        print(f"어깨 높이 차이: {analysis['shoulder']['height_difference']:.3f}")
        print(f"골반 높이 차이: {analysis['pelvic']['height_difference']:.3f}")
        
        # 문제점 표시 (PostureAnalyzer의 실제 반환값에 맞게 수정)
        issues = []
        if analysis['neck']['grade'] in ['C', 'D']:
            issues.append("거북목")
        if analysis['spine']['is_hunched']:
            issues.append("척추 굴곡")
        if analysis['shoulder']['is_asymmetric']:
            issues.append("어깨 비대칭")
        if analysis['pelvic']['is_tilted']:
            issues.append("골반 기울기")
            
        if issues:
            print(f"발견된 문제점: {', '.join(issues)}")
        else:
            print("발견된 문제점: 없음")
            
        print("="*50)
        
    def draw_info_realtime(self, frame, frame_count):
        """실시간 정보 화면에 표시 - 영상 바깥에 콘솔 영역 추가"""
        h, w, _ = frame.shape
        
        # 영상 크기 확장 (오른쪽에 콘솔 영역 추가)
        console_width = 400
        expanded_frame = np.ones((h, w + console_width, 3), dtype=np.uint8) * 255  # 흰색 배경
        
        # 원본 영상을 왼쪽에 배치
        expanded_frame[:, :w] = frame
        
        # 콘솔 영역에 정보 표시
        console_x = w + 20  # 영상 오른쪽 여백
        
        # 제목
        frame = put_korean_text(expanded_frame, "=== 자세 등급 데이터 수집 ===", (console_x, 30), 18, (0, 0, 0))
        
        # 시점 정보
        view_names = {'1': '정면', '2': '측면', '3': '기타 상태'}
        view_name = view_names.get(self.current_view, '알 수 없음')
        frame = put_korean_text(expanded_frame, f"시점: {view_name} ({self.current_view})", 
                               (console_x, 70), 16, (0, 0, 0))
        
        # 자세 등급 정보
        pose_names = {'a': 'A등급 (완벽)', 'b': 'B등급 (양호)', 'c': 'C등급 (보통)', 
                     'd': 'D등급 (나쁨)', 'e': '특수 자세'}
        
        # 등급별 색상 설정
        if auto_grade in ['a', 'b']:
            pose_color = (0, 128, 0)  # 진한 녹색
        elif auto_grade == 'c':
            pose_color = (0, 100, 200)  # 진한 주황색
        elif auto_grade == 'd':
            pose_color = (0, 0, 200)  # 진한 빨간색
        else:  # e등급
            pose_color = (200, 0, 200)  # 진한 마젠타
        
        # 등급 표시
        if self.auto_grade_mode and auto_grade:
            grade_text = f"자세: {pose_names.get(auto_grade, '알 수 없음')} (자동)"
        else:
            grade_text = f"자세: {pose_names.get(auto_grade, '알 수 없음')} (기본값)"
        
        frame = put_korean_text(expanded_frame, grade_text, (console_x, 100), 16, pose_color)
        
        # 자동 등급 모드 상태 표시
        mode_text = "자동 등급 판별: 활성화" if self.auto_grade_mode else "자동 등급 판별: 비활성화"
        mode_color = (0, 128, 0) if self.auto_grade_mode else (128, 128, 128)
        frame = put_korean_text(expanded_frame, mode_text, (console_x, 130), 14, mode_color)
        
        # 프레임 정보
        frame = put_korean_text(expanded_frame, f"프레임: {frame_count}", (console_x, 160), 14, (0, 0, 0))
        
        # 분석 빈도 정보
        analysis_freq = "10프레임마다 분석"
        frame = put_korean_text(expanded_frame, analysis_freq, (console_x, 190), 14, (0, 0, 0))
        
        # 저장 상태 (비동기 처리 표시)
        if frame_count % 10 == 0:
            save_text = "저장 중... (비동기)"
            save_color = (0, 128, 0)
        else:
            save_text = "대기 중... (다음 분석까지)"
            save_color = (128, 128, 128)
        
        frame = put_korean_text(expanded_frame, save_text, (console_x, 220), 14, save_color)
        
        # 큐 상태 표시
        queue_size = self.save_queue.qsize()
        queue_text = f"저장 대기: {queue_size}개"
        frame = put_korean_text(expanded_frame, queue_text, (console_x, 250), 12, (0, 0, 0))
        
        # 키보드 안내
        y_offset = 280
        frame = put_korean_text(expanded_frame, "[키보드 안내]", (console_x, y_offset), 14, (0, 0, 0))
        y_offset += 25
        frame = put_korean_text(expanded_frame, "1,2,3: 시점 변경", (console_x, y_offset), 12, (0, 0, 0))
        y_offset += 20
        frame = put_korean_text(expanded_frame, "auto: 자동 등급 토글", (console_x, y_offset), 12, (0, 0, 0))
        y_offset += 20
        frame = put_korean_text(expanded_frame, "q: 종료", (console_x, y_offset), 12, (0, 0, 0))
        
        return expanded_frame

    def load_grade_model(self):
        """훈련된 등급 판별 모델 로드"""
        # 여러 가능한 경로에서 모델 파일 찾기
        possible_paths = [
            "pose_grade_model.pkl",
            "test_grade/ml_models/pose_grade_model.pkl",
            "../ml_models/pose_grade_model.pkl",
            os.path.join(os.path.dirname(__file__), "../ml_models/pose_grade_model.pkl")
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path:
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.grade_model = model_data['model']
                    self.feature_names = model_data['feature_names']
                print(f"등급 판별 모델 로드 완료: {model_path}")
            except Exception as e:
                print(f"모델 로드 실패: {e}")
                self.grade_model = None
        else:
            print("모델 파일을 찾을 수 없습니다. 다음 경로들을 확인했습니다:")
            for path in possible_paths:
                print(f"  - {path}")
            print("자동 등급 판별 기능을 사용하려면 먼저 모델을 훈련하세요.")
        
    def predict_grade(self, analysis):
        """자동 등급 판별 - 개선된 버전"""
        if self.grade_model is None:
            return None, None
            
        try:
            # 특성 추출 (훈련 시와 동일한 방식)
            basic_features = [
                analysis['neck']['neck_angle'],
                analysis['spine']['spine_angle'],
                analysis['shoulder']['height_difference'],
                analysis['pelvic']['height_difference']
            ]
            
            # 시점 정보 (원-핫 인코딩)
            view_features = [1 if self.current_view == '1' else 0,  # 정면
                           1 if self.current_view == '2' else 0]   # 측면
            
            # 목 관련 추가 특성
            neck_features = [
                analysis['neck']['vertical_deviation'],
                analysis['neck']['neck_angle']
            ]
            
            # 척추 관련 추가 특성
            spine_features = [
                analysis['spine']['spine_angle'],
                1 if analysis['spine']['is_hunched'] else 0
            ]
            
            # 어깨 관련 추가 특성
            shoulder_features = [
                analysis['shoulder']['height_difference'],
                analysis['shoulder']['shoulder_angle'],
                1 if analysis['shoulder']['is_asymmetric'] else 0
            ]
            
            # 골반 관련 추가 특성
            pelvic_features = [
                analysis['pelvic']['height_difference'],
                analysis['pelvic']['pelvic_angle'],
                1 if analysis['pelvic']['is_tilted'] else 0
            ]
            
            # 새로운 파생 특성들 (훈련 시와 동일)
            derived_features = [
                # 목과 척추의 상호작용
                abs(analysis['neck']['neck_angle']) * abs(analysis['spine']['spine_angle']),
                
                # 어깨와 골반의 비대칭성 합계
                analysis['shoulder']['height_difference'] + analysis['pelvic']['height_difference'],
                
                # 전체 자세 점수 (간단한 계산)
                100 - (abs(analysis['neck']['neck_angle']) * 2 + 
                       abs(analysis['spine']['spine_angle']) * 2 +
                       analysis['shoulder']['height_difference'] * 1000 +
                       analysis['pelvic']['height_difference'] * 1000),
                
                # 목 각도의 제곱 (비선형 관계)
                analysis['neck']['neck_angle'] ** 2,
                
                # 척추 각도의 제곱
                analysis['spine']['spine_angle'] ** 2,
                
                # 어깨 비대칭성의 제곱
                analysis['shoulder']['height_difference'] ** 2,
                
                # 골반 기울기의 제곱
                analysis['pelvic']['height_difference'] ** 2,
                
                # 문제점 개수
                (1 if analysis['spine']['is_hunched'] else 0) +
                (1 if analysis['shoulder']['is_asymmetric'] else 0) +
                (1 if analysis['pelvic']['is_tilted'] else 0) +
                (1 if analysis['neck']['grade'] in ['C', 'D'] else 0)
            ]
            
            # 모든 특성 결합
            features = (basic_features + view_features + neck_features + 
                       spine_features + shoulder_features + pelvic_features + derived_features)
            
            # 예측 수행
            prediction = self.grade_model.predict([features])[0]
            
            # 종합 점수 계산 (개선된 방식)
            total_score = self.calculate_total_score(analysis)
            
            return prediction, total_score
            
        except Exception as e:
            print(f"등급 예측 중 오류: {e}")
            return None, None
            
    def calculate_total_score(self, analysis):
        """종합 점수 계산"""
        score = 100
        
        # 목 각도에 따른 점수 차감
        neck_angle = abs(analysis['neck']['neck_angle'])
        if neck_angle > 15:
            score -= 20
        elif neck_angle > 10:
            score -= 10
        elif neck_angle > 5:
            score -= 5
            
        # 척추 각도에 따른 점수 차감
        spine_angle = abs(analysis['spine']['spine_angle'])
        if spine_angle > 10:
            score -= 20
        elif spine_angle > 5:
            score -= 10
            
        # 어깨 비대칭에 따른 점수 차감
        shoulder_diff = analysis['shoulder']['height_difference']
        if shoulder_diff > 0.05:
            score -= 15
        elif shoulder_diff > 0.02:
            score -= 8
            
        # 골반 기울기에 따른 점수 차감
        pelvic_diff = analysis['pelvic']['height_difference']
        if pelvic_diff > 0.05:
            score -= 15
        elif pelvic_diff > 0.02:
            score -= 8
            
        return max(0, score)

def main():
    collector = PoseGradeCollector()
    
    print("=== 자세 등급 데이터 수집 시스템 ===")
    print("실시간 키보드 입력 방식으로 데이터를 수집합니다.")
    print("'auto'를 입력하면 자동 등급 판별 모드를 활성화할 수 있습니다.")
    print("분석은 10프레임마다 수행됩니다.")
    print("데이터 수집과 저장이 동시에 실행됩니다.")
    
    participant_id = input("참가자 ID를 입력하세요: ")
    
    # 실시간 키보드 입력 방식만 실행
    collector.collect_data_realtime(participant_id)
    
    print("\n데이터 수집이 완료되었습니다!")

if __name__ == "__main__":
    main() 