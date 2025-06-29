from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
import numpy as np
import cv2
from .neck import PostureAnalyzer
from .models import User, PostureRecord, db
import mediapipe as mp
from datetime import datetime, timedelta
import time

crud = Blueprint(
    'crud',
    __name__,
    template_folder='templates',
    static_folder='static'
)

analyzer = PostureAnalyzer()

# 상태 관리 클래스
class PoseStateManager:
    def __init__(self):
        self.state = "no_human_detected"
        self.state_start_time = time.time()
        self.last_state_change = time.time()
        self.front_pose_frames = []  # 20프레임 저장용
        self.front_pose_area = None
        self.prev_landmarks = None
        self.front_pose_stable_start = None
        self.no_landmark_start = None
        self.SIDE_RATIO = 0.7  # 측면 면적 비율 기준
        self.STABLE_DURATION = 2.0  # 정면 안정화 시간
        self.MOVE_THRESHOLD = 0.02  # landmark 이동량 임계값
        self.NO_LANDMARK_TIMEOUT = 10.0  # 관절 미감지 10초

    def update_state(self, landmarks):
        current_time = time.time()
        # 상태 1: no_human_detected
        if self.state == "no_human_detected":
            if landmarks is not None:
                print("[전이] no_human_detected → detecting_front_pose")
                self.state = "detecting_front_pose"
                self.state_start_time = current_time
                self.front_pose_frames = []
                self.prev_landmarks = None
                self.front_pose_stable_start = None
                self.no_landmark_start = None
        # 상태 2: detecting_front_pose
        elif self.state == "detecting_front_pose":
            if landmarks is None:
                print("[전이] detecting_front_pose → no_human_detected")
                self.state = "no_human_detected"
                self.state_start_time = current_time
                self.front_pose_frames = []
                self.prev_landmarks = None
                self.front_pose_stable_start = None
                self.no_landmark_start = current_time
                return
            # 어깨, 귀, 코 좌표만 추출
            key_indices = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
                           mp.solutions.pose.PoseLandmark.LEFT_EAR.value, mp.solutions.pose.PoseLandmark.RIGHT_EAR.value,
                           mp.solutions.pose.PoseLandmark.NOSE.value]
            keypoints = np.array([[landmarks[i].x, landmarks[i].y] for i in key_indices])
            # 20프레임 저장
            self.front_pose_frames.append(keypoints)
            if len(self.front_pose_frames) > 20:
                self.front_pose_frames.pop(0)
            # 이동량 계산
            if self.prev_landmarks is not None:
                move = np.linalg.norm(keypoints - self.prev_landmarks, axis=1).mean()
            else:
                move = 0
            self.prev_landmarks = keypoints
            # 안정화 시작 체크
            if move < self.MOVE_THRESHOLD:
                if self.front_pose_stable_start is None:
                    self.front_pose_stable_start = current_time
                elif current_time - self.front_pose_stable_start >= self.STABLE_DURATION:
                    # 평균 면적 계산 (어깨-귀 사각형 넓이)
                    arr = np.array(self.front_pose_frames)
                    left_shoulder = arr[:,0]
                    right_shoulder = arr[:,1]
                    left_ear = arr[:,2]
                    right_ear = arr[:,3]
                    # 어깨-귀 사각형 넓이(대략적)
                    width = np.linalg.norm(left_shoulder - right_shoulder, axis=1).mean()
                    height = np.linalg.norm(left_ear - left_shoulder, axis=1).mean()
                    area = width * height
                    self.front_pose_area = area
                    print(f"[전이] detecting_front_pose → waiting_side_pose (정면 안정화, 면적:{area:.4f})")
                    self.state = "waiting_side_pose"
                    self.state_start_time = current_time
                    self.last_state_change = current_time
                    self.front_pose_frames = []
                    self.prev_landmarks = None
                    self.front_pose_stable_start = None
            else:
                self.front_pose_stable_start = None
        # 상태 3: waiting_side_pose
        elif self.state == "waiting_side_pose":
            if landmarks is None:
                if self.no_landmark_start is None:
                    self.no_landmark_start = current_time
                elif current_time - self.no_landmark_start > self.NO_LANDMARK_TIMEOUT:
                    print("[전이] waiting_side_pose → no_human_detected (관절 미감지 10초)")
                    self.state = "no_human_detected"
                    self.state_start_time = current_time
                    self.front_pose_area = None
                    self.no_landmark_start = None
                return
            else:
                self.no_landmark_start = None
            # 측면 판별: 정면 면적 대비 70% 이하로 줄어들면
            key_indices = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
                           mp.solutions.pose.PoseLandmark.LEFT_EAR.value, mp.solutions.pose.PoseLandmark.RIGHT_EAR.value]
            keypoints = np.array([[landmarks[i].x, landmarks[i].y] for i in key_indices])
            width = np.linalg.norm(keypoints[0] - keypoints[1])
            height = np.linalg.norm(keypoints[2] - keypoints[0])
            area = width * height
            if self.front_pose_area and area < self.front_pose_area * self.SIDE_RATIO:
                print(f"[전이] waiting_side_pose → analyzing_side_pose (측면 감지, 면적:{area:.4f})")
                self.state = "analyzing_side_pose"
                self.state_start_time = current_time
                self.last_state_change = current_time
        # 상태 4: analyzing_side_pose
        elif self.state == "analyzing_side_pose":
            if landmarks is None:
                print("[전이] analyzing_side_pose → no_human_detected (관절 미감지)")
                self.state = "no_human_detected"
                self.state_start_time = current_time
                self.front_pose_area = None
                self.no_landmark_start = current_time
            # 분석 동작은 외부에서 수행

    def get_state_message(self):
        if self.state == "no_human_detected":
            return "카메라 앞에 앉아주세요"
        elif self.state == "detecting_front_pose":
            return "정면 자세 측정을 시작합니다."
        elif self.state == "waiting_side_pose":
            return "카메라에 왼쪽 또는 오른쪽 측면을 보이고 앉아주세요"
        elif self.state == "analyzing_side_pose":
            return "바른자세 측정을 시작합니다. 학습을 시작하세요."
        return "알 수 없는 상태"

# 전역 상태 관리자 인스턴스
state_manager = PoseStateManager()

def login_required(f):
    """로그인 필요 데코레이터"""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('로그인이 필요합니다.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@crud.route('/')
@login_required
def index():
    user = User.query.get(session['user_id'])
    return render_template('crud/index.html', user=user)

@crud.route('/analyze', methods=['POST'])
@login_required
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
            
            # [AI 수정] 상체 랜드마크만 추출 (얼굴 제외, 귀부터 시작)
            upper_body_landmarks = []
            for i, landmark in enumerate(lm):
                if i >= 7:  # 귀부터 시작 (7: LEFT_EAR, 8: RIGHT_EAR, 11: 어깨)
                    upper_body_landmarks.append({'x': landmark.x, 'y': landmark.y, 'index': i})
            
            # 상태 업데이트
            state_manager.update_state(lm)
            
            # 분석 중일 때만 자세 분석 실행
            if state_manager.state == "analyzing_side_pose":
                # 자세 분석 수행
                neck_result = analyzer.analyze_turtle_neck_detailed(lm)
                spine_result = analyzer.analyze_spine_curvature(lm)
                shoulder_result = analyzer.analyze_shoulder_asymmetry(lm)
                pelvic_result = analyzer.analyze_pelvic_tilt(lm)
                twist_result = analyzer.analyze_spine_twisting(lm)
                
                # 데이터베이스에 분석 결과 저장
                user = User.query.get(session['user_id'])
                posture_record = PostureRecord(
                    user_id=user.id,
                    neck_angle=neck_result['neck_angle'],
                    neck_grade=neck_result['grade'],
                    neck_description=neck_result['grade_description'],
                    spine_is_hunched=spine_result['is_hunched'],
                    spine_angle=spine_result['spine_angle'],
                    shoulder_is_asymmetric=shoulder_result['is_asymmetric'],
                    shoulder_height_difference=shoulder_result['height_difference'],
                    pelvic_is_tilted=pelvic_result['is_tilted'],
                    pelvic_angle=pelvic_result['pelvic_angle'],
                    spine_is_twisted=twist_result['is_twisted'],
                    spine_alignment=twist_result['spine_alignment']
                )
                
                # 종합 점수 계산
                posture_record.overall_score = posture_record.calculate_overall_score()
                posture_record.overall_grade = posture_record.calculate_overall_grade()
                
                try:
                    db.session.add(posture_record)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print(f"데이터베이스 저장 오류: {e}")
                
                return jsonify({
                        'landmarks': upper_body_landmarks,  # [AI 추가] 상체 랜드마크 반환
                        'state': state_manager.state,
                        'state_message': state_manager.get_state_message(),
                    'neck': neck_result,
                    'spine': spine_result,
                    'shoulder': shoulder_result,
                    'pelvic': pelvic_result,
                    'twist': twist_result,
                    'overall_score': posture_record.overall_score,
                    'overall_grade': posture_record.overall_grade
                })
            else:
                # 분석 중이 아닐 때는 상태 정보만 반환
                stable_time = None
                if state_manager.state == "detecting_front_pose" and state_manager.front_pose_stable_start:
                    stable_time = time.time() - state_manager.front_pose_stable_start
                
                return jsonify({
                    'landmarks': upper_body_landmarks,  # [AI 추가] 상체 랜드마크 반환
                    'state': state_manager.state,
                    'state_message': state_manager.get_state_message(),
                    'stable_time': stable_time
                })
        else:
            return jsonify({'error': 'No person detected'})

@crud.route('/history')
@login_required
def history():
    """사용자의 자세 분석 기록 페이지"""
    user = User.query.get(session['user_id'])
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # 페이지네이션으로 기록 가져오기
    records = user.posture_records.order_by(
        PostureRecord.analysis_date.desc()
    ).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('crud/history.html', 
                         user=user, 
                         records=records)

@crud.route('/statistics')
@login_required
def statistics():
    """사용자의 자세 분석 통계 페이지"""
    user = User.query.get(session['user_id'])
    
    # 전체 통계
    total_records = user.posture_records.count()
    if total_records > 0:
        all_records = user.posture_records.all()
        avg_score = sum(record.calculate_overall_score() for record in all_records) / total_records
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for record in all_records:
            grade = record.calculate_overall_grade()
            grade_counts[grade] += 1
        week_ago = datetime.now() - timedelta(days=7)
        recent_records = [r for r in all_records if r.analysis_date >= week_ago]
        recent_avg = sum(r.calculate_overall_score() for r in recent_records) / len(recent_records) if recent_records else 0
        monthly_stats = {}
        for i in range(6):
            month_start = datetime.now().replace(day=1) - timedelta(days=30*i)
            month_end = month_start.replace(day=28) + timedelta(days=4)
            month_end = month_end.replace(day=1) - timedelta(days=1)
            month_records = [r for r in all_records if month_start <= r.analysis_date <= month_end]
            if month_records:
                monthly_stats[month_start.strftime('%Y-%m')] = {
                    'count': len(month_records),
                    'avg_score': sum(r.calculate_overall_score() for r in month_records) / len(month_records)
                }
    else:
        avg_score = 0
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        recent_avg = 0
        monthly_stats = {}
    
    return render_template('crud/statistics.html',
                         user=user,
                         total_records=total_records,
                         avg_score=avg_score,
                         grade_counts=grade_counts,
                         recent_avg=recent_avg,
                         monthly_stats=monthly_stats,
                         now=datetime.now())
