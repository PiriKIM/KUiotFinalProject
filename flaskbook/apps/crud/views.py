from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
import numpy as np
import cv2
from .neck import PostureAnalyzer
from .models import User, PostureRecord, db
import mediapipe as mp
from datetime import datetime, timedelta

crud = Blueprint(
    'crud',
    __name__,
    template_folder='templates',
    static_folder='static'
)

analyzer = PostureAnalyzer()

@crud.route('/')
@login_required
def index():
    user = current_user
    return render_template('crud/index.html', user=user)

@crud.route('/realtime')
@login_required
def realtime():
    """실시간 분석 페이지로 리다이렉트"""
    return redirect(url_for('realtime.realtime_analysis_page'))

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
            
            # 자세 분석 수행
            neck_result = analyzer.analyze_turtle_neck_detailed(lm)
            spine_result = analyzer.analyze_spine_curvature(lm)
            shoulder_result = analyzer.analyze_shoulder_asymmetry(lm)
            pelvic_result = analyzer.analyze_pelvic_tilt(lm)
            twist_result = analyzer.analyze_spine_twisting(lm)
            
            # 데이터베이스에 분석 결과 저장
            user = current_user
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
                'neck': neck_result,
                'spine': spine_result,
                'shoulder': shoulder_result,
                'pelvic': pelvic_result,
                'twist': twist_result,
                'overall_score': posture_record.overall_score,
                'overall_grade': posture_record.overall_grade
            })
        else:
            return jsonify({'error': 'No person detected'})

@crud.route('/history')
@login_required
def history():
    """사용자의 실시간 자세 분석 기록 페이지"""
    user = current_user
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # 실시간 분석 기록을 페이지네이션으로 가져오기
    from .models import RealtimePostureRecord
    records = user.realtime_records.order_by(
        RealtimePostureRecord.timestamp.desc()
    ).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('crud/history.html', 
                         user=user, 
                         records=records)

@crud.route('/statistics')
@login_required
def statistics():
    """사용자의 실시간 자세 분석 통계 페이지"""
    user = current_user
    
    # 실시간 분석 통계
    from .models import RealtimePostureRecord
    total_records = user.realtime_records.count()
    
    if total_records > 0:
        all_records = user.realtime_records.all()
        
        # 등급별 통계 (N/A 제외)
        grade_counts = {'A': 0, 'B': 0, 'C': 0}
        for record in all_records:
            if record.posture_grade and record.posture_grade != 'N/A':
                grade_counts[record.posture_grade] += 1
        
        # 평균 CVA 각도
        valid_angles = [r.cva_angle for r in all_records if r.cva_angle is not None]
        avg_cva_angle = sum(valid_angles) / len(valid_angles) if valid_angles else 0
        
        # 최근 7일 통계 (N/A 제외)
        week_ago = datetime.now() - timedelta(days=7)
        recent_records = [r for r in all_records if r.timestamp >= week_ago]
        recent_grade_counts = {'A': 0, 'B': 0, 'C': 0}
        for record in recent_records:
            if record.posture_grade and record.posture_grade != 'N/A':
                recent_grade_counts[record.posture_grade] += 1
        
        # 측면별 통계
        side_counts = {}
        for record in all_records:
            side = record.detected_side if record.detected_side else 'unknown'
            side_counts[side] = side_counts.get(side, 0) + 1
        
        # 월별 통계
        monthly_stats = {}
        for i in range(6):
            month_start = datetime.now().replace(day=1) - timedelta(days=30*i)
            month_end = month_start.replace(day=28) + timedelta(days=4)
            month_end = month_end.replace(day=1) - timedelta(days=1)
            month_records = [r for r in all_records if month_start <= r.timestamp <= month_end]
            if month_records:
                month_grade_counts = {'A': 0, 'B': 0, 'C': 0}
                for record in month_records:
                    if record.posture_grade and record.posture_grade != 'N/A':
                        month_grade_counts[record.posture_grade] += 1
                
                monthly_stats[month_start.strftime('%Y-%m')] = {
                    'count': len(month_records),
                    'grade_counts': month_grade_counts
                }
    else:
        grade_counts = {'A': 0, 'B': 0, 'C': 0}
        avg_cva_angle = 0
        recent_grade_counts = {'A': 0, 'B': 0, 'C': 0}
        side_counts = {}
        monthly_stats = {}
    
    return render_template('crud/statistics.html',
                         user=user,
                         total_records=total_records,
                         grade_counts=grade_counts,
                         avg_cva_angle=avg_cva_angle,
                         recent_grade_counts=recent_grade_counts,
                         side_counts=side_counts,
                         monthly_stats=monthly_stats,
                         now=datetime.now())
