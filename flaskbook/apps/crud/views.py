from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash
import numpy as np
import cv2
from .neck import PostureAnalyzer
from .models import User, PostureRecord, db
import mediapipe as mp
from datetime import datetime, timedelta
import time
import subprocess
import threading


crud = Blueprint(
    'crud',
    __name__,
    template_folder='templates',
    static_folder='static'
)

analyzer = PostureAnalyzer()

# ESP32 부저 제어 설정
ESP32_IP = "192.168.0.102"  # ESP32 IP 주소
ESP32_PORT = 81
ESP32_BUZZER_URL = f"http://{ESP32_IP}:{ESP32_PORT}/buzzer"

# 상태 관리 클래스 (ESP32-CAM 전용으로 단순화)
class ESP32PoseStateManager:
    def __init__(self):
        self.state = "waiting_for_connection"
        self.state_start_time = time.time()
        self.last_state_change = time.time()
        self.front_pose_frames = []
        self.front_pose_area = None
        self.prev_landmarks = None
        self.front_pose_stable_start = None
        self.STABLE_DURATION = 2.0  # 정면 안정화 시간
        self.MOVE_THRESHOLD = 0.02  # landmark 이동량 임계값

    def update_state(self, landmarks):
        current_time = time.time()
        
        # 상태 1: waiting_for_connection
        if self.state == "waiting_for_connection":
            if landmarks is not None:
                print("[전이] waiting_for_connection → detecting_front_pose")
                self.state = "detecting_front_pose"
                self.state_start_time = current_time
                self.front_pose_frames = []
                self.prev_landmarks = None
                self.front_pose_stable_start = None
        
        # 상태 2: detecting_front_pose
        elif self.state == "detecting_front_pose":
            if landmarks is None:
                print("[전이] detecting_front_pose → waiting_for_connection")
                self.state = "waiting_for_connection"
                self.state_start_time = current_time
                self.front_pose_frames = []
                self.prev_landmarks = None
                self.front_pose_stable_start = None
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
                print("[전이] waiting_side_pose → waiting_for_connection")
                self.state = "waiting_for_connection"
                self.state_start_time = current_time
                self.front_pose_area = None
                return
            
            # 측면 판별: 정면 면적 대비 70% 이하로 줄어들면
            key_indices = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
                           mp.solutions.pose.PoseLandmark.LEFT_EAR.value, mp.solutions.pose.PoseLandmark.RIGHT_EAR.value]
            keypoints = np.array([[landmarks[i].x, landmarks[i].y] for i in key_indices])
            width = np.linalg.norm(keypoints[0] - keypoints[1])
            height = np.linalg.norm(keypoints[2] - keypoints[0])
            area = width * height
            
            if self.front_pose_area and area < self.front_pose_area * 0.7:
                print(f"[전이] waiting_side_pose → analyzing_side_pose (측면 감지, 면적:{area:.4f})")
                self.state = "analyzing_side_pose"
                self.state_start_time = current_time
                self.last_state_change = current_time
        
        # 상태 4: analyzing_side_pose
        elif self.state == "analyzing_side_pose":
            if landmarks is None:
                print("[전이] analyzing_side_pose → waiting_for_connection")
                self.state = "waiting_for_connection"
                self.state_start_time = current_time
                self.front_pose_area = None

    def get_state_message(self):
        if self.state == "waiting_for_connection":
            return "ESP32-CAM 연결을 기다리는 중..."
        elif self.state == "detecting_front_pose":
            return "정면 자세 측정을 시작합니다."
        elif self.state == "waiting_side_pose":
            return "카메라에 왼쪽 또는 오른쪽 측면을 보이고 앉아주세요"
        elif self.state == "analyzing_side_pose":
            return "바른자세 측정을 시작합니다. 학습을 시작하세요."
        return "알 수 없는 상태"

# 전역 상태 관리자 인스턴스
state_manager = ESP32PoseStateManager()

# 자세 분석 상태 관리
class PostureBuzzerManager:
    def __init__(self):
        self.bad_posture_count = 0
        self.good_posture_count = 0
        self.last_buzzer_time = 0
        self.buzzer_cooldown = 5  # 5초 쿨다운
        self.bad_posture_threshold = 3  # 3번 연속 나쁜 자세 감지시 부저
        self.good_posture_reset = 5  # 5번 연속 좋은 자세시 카운터 리셋
        self.buzzer_enabled = True  # 부저 기능 활성화/비활성화
        self.esp32_connected = False  # ESP32 연결 상태
        self.last_connection_check = 0  # 마지막 연결 확인 시간
        self.connection_check_interval = 3  # 3초마다 연결 확인 (더 빠른 응답)
    
    def trigger_buzzer(self, action='trigger', volume=None):
        """ESP32 부저 제어 (trigger 및 볼륨 조정 지원)"""
        if not self.buzzer_enabled:
            print(f"🔕 부저 비활성화 상태: {action} 명령 무시")
            return False
        
        # URL 구성
        if action == 'volume' and volume is not None:
            url = f"{ESP32_BUZZER_URL}?action={action}&value={volume}"
            print(f"🌐 ESP32 볼륨 요청: {url}")
        elif action == 'trigger':
            url = f"{ESP32_BUZZER_URL}?action={action}"
            print(f"🌐 ESP32 부저 울림 요청: {url}")
        else:
            print(f"🔕 지원하지 않는 명령: {action}")
            return False
        
        # curl 명령어로 직접 요청 (재시도 로직 포함)
        import subprocess
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 ESP32 요청 시도 {attempt + 1}/{max_retries}")
                start_time = time.time()
                
                # curl 명령어 구성
                cmd = ['curl', '-s', '--connect-timeout', '5', url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                end_time = time.time()
                
                if result.returncode == 0 and result.stdout:
                    print(f"📡 ESP32 응답: 200 - {result.stdout.strip()} (응답시간: {end_time-start_time:.2f}초)")
                    if action == 'volume':
                        print(f"🔔 ESP32 볼륨 설정 성공: {volume}%")
                    else:
                        print(f"🔔 ESP32 부저 울림 성공!")
                    return True
                else:
                    print(f"❌ ESP32 요청 실패: {result.stderr}")
                    if attempt < max_retries - 1:
                        print(f"⏳ 1초 후 재시도...")
                        time.sleep(1)
                    continue
                    
            except subprocess.TimeoutExpired:
                print(f"❌ ESP32 요청 타임아웃 (8초) - 시도 {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    print(f"⏳ 2초 후 재시도...")
                    time.sleep(2)
                continue
            except Exception as e:
                print(f"❌ ESP32 요청 오류: {e} - 시도 {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    print(f"⏳ 1초 후 재시도...")
                    time.sleep(1)
                continue
        
        print(f"❌ ESP32 요청 최종 실패 ({max_retries}회 시도)")
        return False
    
    def trigger_buzzer_async(self, action='trigger', volume=None):
        """비동기 부저 제어 (제거됨)"""
        # 부저 기능 제거됨
        print(f"🔔 부저 제어 호출됨: action={action}, volume={volume} (기능 제거됨)")
        return False
    
    def check_esp32_connection(self):
        """ESP32 연결 상태 확인 (제거됨)"""
        current_time = time.time()
        if current_time - self.last_connection_check < self.connection_check_interval:
            return self.esp32_connected  # 캐시된 상태 반환
        
        # ESP32 연결 확인 기능 제거됨
        self.esp32_connected = False
        self.last_connection_check = current_time
        return self.esp32_connected
    
    def check_posture_and_buzzer(self, posture_result):
        """자세 분석 결과에 따른 부저 제어 (등급별)"""
        if not self.buzzer_enabled:
            return False
            
        current_grade = posture_result.get('grade', 'A')
        current_score = posture_result.get('score', 100)
        
        # 이전 등급과 비교
        if not hasattr(self, 'last_grade'):
            self.last_grade = current_grade
            self.last_score = current_score
            return False
        
        # 등급이 바뀌었는지 확인
        grade_changed = (self.last_grade != current_grade)
        score_dropped = (self.last_score - current_score > 10)  # 10점 이상 하락
        
        # 등급 업데이트
        self.last_grade = current_grade
        self.last_score = current_score
        
        # 등급별 부저 울림 조건 (더 적극적으로)
        if current_grade in ['C', 'D', 'F'] or score_dropped:
            # ESP32 연결 상태 확인
            if not self.check_esp32_connection():
                print("⚠️ ESP32 연결되지 않음 - 부저 제어 건너뜀")
                return False
            
            print(f"🚨 나쁜 자세 등급 감지: {current_grade} (점수: {current_score})")
            self.trigger_buzzer_async('trigger')  # 비동기로 변경
            return True
        elif current_grade == 'B' and (grade_changed or score_dropped):
            # ESP32 연결 상태 확인
            if not self.check_esp32_connection():
                print("⚠️ ESP32 연결되지 않음 - 부저 제어 건너뜀")
                return False
            
            print(f"⚠️ 주의 자세 등급: {current_grade} (점수: {current_score})")
            self.trigger_buzzer_async('trigger')  # 비동기로 변경
            return True
        elif current_grade == 'A' and grade_changed:
            print(f"✅ 좋은 자세로 개선: {current_grade} (점수: {current_score})")
            # A등급으로 개선된 경우는 부저 울리지 않음
            return False
        
        return False

# 전역 부저 관리자 인스턴스
buzzer_manager = PostureBuzzerManager()

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
            
            # 상체 랜드마크만 추출 (얼굴 제외, 귀부터 시작)
            upper_body_landmarks = []
            for i, landmark in enumerate(lm):
                if i >= 7:  # 귀부터 시작 (7: LEFT_EAR, 8: RIGHT_EAR, 11: 어깨)
                    upper_body_landmarks.append({'x': landmark.x, 'y': landmark.y, 'index': i})
            
            # 상태 업데이트
            state_manager.update_state(lm)
            print(f"현재 상태: {state_manager.state}")
            
            # 분석 중일 때만 자세 분석 실행
            if state_manager.state == "analyzing_side_pose":
                print("측면 자세 분석 시작")
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
                
                print(f"계산된 점수: {posture_record.overall_score}, 등급: {posture_record.overall_grade}")
                
                # 자세 등급에 따른 부저 제어
                posture_result = {
                    'grade': posture_record.overall_grade,
                    'neck_angle': neck_result['neck_angle'],
                    'score': posture_record.overall_score
                }
                buzzer_triggered = buzzer_manager.check_posture_and_buzzer(posture_result)
                
                try:
                    db.session.add(posture_record)
                    db.session.commit()
                    print("데이터베이스 저장 성공")
                except Exception as e:
                    db.session.rollback()
                    print(f"데이터베이스 저장 오류: {e}")
                
                response_data = {
                    'landmarks': upper_body_landmarks,
                    'state': state_manager.state,
                    'state_message': state_manager.get_state_message(),
                    'neck': neck_result,
                    'spine': spine_result,
                    'shoulder': shoulder_result,
                    'pelvic': pelvic_result,
                    'twist': twist_result,
                    'overall_score': posture_record.overall_score,
                    'overall_grade': posture_record.overall_grade,
                    'buzzer_triggered': buzzer_triggered,
                    'bad_posture_count': buzzer_manager.bad_posture_count,
                    'good_posture_count': buzzer_manager.good_posture_count
                }
                print(f"응답 데이터: {response_data}")
                return jsonify(response_data)
            else:
                # 분석 중이 아닐 때는 상태 정보만 반환
                stable_time = None
                if state_manager.state == "detecting_front_pose" and state_manager.front_pose_stable_start:
                    stable_time = time.time() - state_manager.front_pose_stable_start
                
                return jsonify({
                    'landmarks': upper_body_landmarks,
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

@crud.route('/buzzer-test')
@login_required
def buzzer_test():
    """부저 테스트 페이지"""
    return render_template('crud/buzzer_test.html')

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
                         monthly_stats=monthly_stats)

@crud.route('/api/buzzer', methods=['GET', 'POST'])
@login_required
def buzzer_control():
    """ESP32 부저 제어 API"""
    action = request.args.get('action', 'status')
    
    if action == 'status':
        return jsonify({
            'enabled': buzzer_manager.buzzer_enabled,
            'bad_posture_count': buzzer_manager.bad_posture_count,
            'good_posture_count': buzzer_manager.good_posture_count,
            'threshold': buzzer_manager.bad_posture_threshold,
            'cooldown': buzzer_manager.buzzer_cooldown
        })
    
    elif action == 'trigger':
        success = buzzer_manager.trigger_buzzer('trigger')
        return jsonify({'success': success, 'action': 'trigger'})
    
    elif action == 'test':
        success = buzzer_manager.trigger_buzzer('test')
        return jsonify({'success': success, 'action': 'test'})
    
    elif action == 'on':
        success = buzzer_manager.trigger_buzzer('on')
        return jsonify({'success': success, 'action': 'on'})
    
    elif action == 'off':
        success = buzzer_manager.trigger_buzzer('off')
        return jsonify({'success': success, 'action': 'off'})
    
    elif action == 'enable':
        buzzer_manager.buzzer_enabled = True
        return jsonify({'success': True, 'enabled': True})
    
    elif action == 'disable':
        buzzer_manager.buzzer_enabled = False
        return jsonify({'success': True, 'enabled': False})
    
    elif action == 'reset':
        buzzer_manager.bad_posture_count = 0
        buzzer_manager.good_posture_count = 0
        return jsonify({'success': True, 'message': '카운터 리셋됨'})
    
    elif action == 'volume+':
        success = buzzer_manager.trigger_buzzer('volume+')
        return jsonify({'success': success, 'action': 'volume+'})
    
    elif action == 'volume-':
        success = buzzer_manager.trigger_buzzer('volume-')
        return jsonify({'success': success, 'action': 'volume-'})
    
    elif action == 'volume':
        volume = request.args.get('value', type=int)
        if volume is not None:
            success = buzzer_manager.trigger_buzzer('volume', volume)
            return jsonify({'success': success, 'action': 'volume', 'value': volume})
        else:
            return jsonify({'error': 'Volume value required'}), 400
    
    else:
        return jsonify({'error': 'Invalid action'}), 400

@crud.route('/api/buzzer/settings', methods=['POST'])
@login_required
def buzzer_settings():
    """부저 설정 변경 API"""
    data = request.get_json()
    
    if 'threshold' in data:
        buzzer_manager.bad_posture_threshold = int(data['threshold'])
    
    if 'cooldown' in data:
        buzzer_manager.buzzer_cooldown = int(data['cooldown'])
    
    if 'enabled' in data:
        buzzer_manager.buzzer_enabled = bool(data['enabled'])
    
    return jsonify({
        'success': True,
        'threshold': buzzer_manager.bad_posture_threshold,
        'cooldown': buzzer_manager.buzzer_cooldown,
        'enabled': buzzer_manager.buzzer_enabled
    })

@crud.route('/api/buzzer/trigger', methods=['POST'])
@login_required
def trigger_buzzer_now():
    """즉시 부저 울리기 (독립적인 컨트롤러 사용)"""
    try:
        success = buzzer_client.trigger_buzzer()
        return jsonify({
            'status': 'success' if success else 'error',
            'message': '부저 울림 성공!' if success else '부저 울림 실패'
        })
    except Exception as e:
        print(f"❌ 부저 제어 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@crud.route('/api/buzzer/volume', methods=['POST'])
@login_required
def set_buzzer_volume():
    """부저 볼륨 설정 (독립적인 컨트롤러 사용)"""
    try:
        data = request.get_json()
        volume = data.get('volume', 50)
        
        success = buzzer_client.set_volume(volume)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': f'볼륨 {volume}% 설정 완료' if success else '볼륨 설정 실패',
            'volume': volume
        })
    except Exception as e:
        print(f"❌ 볼륨 설정 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500