from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from .models import User, PostureRecord, db
from datetime import datetime

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # 유효성 검사
        if not username or not email or not password:
            flash('모든 필드를 입력해주세요.', 'error')
            return render_template('crud/register.html')
        
        if password != confirm_password:
            flash('비밀번호가 일치하지 않습니다.', 'error')
            return render_template('crud/register.html')
        
        # 중복 검사
        if User.query.filter_by(username=username).first():
            flash('이미 존재하는 사용자명입니다.', 'error')
            return render_template('crud/register.html')
        
        if User.query.filter_by(email=email).first():
            flash('이미 존재하는 이메일입니다.', 'error')
            return render_template('crud/register.html')
        
        # 새 사용자 생성
        new_user = User(
            username=username,
            email=email
        )
        new_user.password = password
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('회원가입이 완료되었습니다. 로그인해주세요.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            flash('회원가입 중 오류가 발생했습니다.', 'error')
            return render_template('crud/register.html')
    
    return render_template('crud/register.html')

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('사용자명과 비밀번호를 입력해주세요.', 'error')
            return render_template('crud/login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            user.update_last_login()
            flash(f'{user.username}님, 환영합니다!', 'success')
            return redirect(url_for('realtime.realtime_analysis_page'))
        else:
            flash('사용자명 또는 비밀번호가 올바르지 않습니다.', 'error')
            return render_template('crud/login.html')
    
    return render_template('crud/login.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('auth.login'))

@auth.route('/profile')
@login_required
def profile():
    user = current_user
    
    # 최근 실시간 자세 분석 기록 가져오기 (최근 10개)
    from apps.crud.models import RealtimePostureRecord
    recent_records = user.realtime_records.order_by(RealtimePostureRecord.timestamp.desc()).limit(10).all()
    
    # 실시간 분석 통계 계산
    total_analyses = user.realtime_records.count()
    if total_analyses > 0:
        all_records = user.realtime_records.all()
        
        # 등급별 통계 (N/A 제외)
        grade_counts = {'A': 0, 'B': 0, 'C': 0}
        for record in all_records:
            if record.posture_grade and record.posture_grade != 'N/A':
                grade_counts[record.posture_grade] += 1
        
        # 평균 CVA 각도
        valid_angles = [r.cva_angle for r in all_records if r.cva_angle is not None]
        avg_cva_angle = sum(valid_angles) / len(valid_angles) if valid_angles else 0
        
        # 측면별 통계
        side_counts = {}
        for record in all_records:
            side = record.detected_side if record.detected_side else 'unknown'
            side_counts[side] = side_counts.get(side, 0) + 1
    else:
        avg_cva_angle = 0
        grade_counts = {'A': 0, 'B': 0, 'C': 0}
        side_counts = {}
    
    return render_template('crud/profile.html', 
                         user=user, 
                         recent_records=recent_records,
                         total_analyses=total_analyses,
                         avg_cva_angle=avg_cva_angle,
                         grade_counts=grade_counts,
                         side_counts=side_counts) 