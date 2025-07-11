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
    """사용자의 자세 분석 기록 페이지"""
    user = current_user
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
    user = current_user
    
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
