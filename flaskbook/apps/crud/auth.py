from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash
from .models import User, db
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
            session['user_id'] = user.id
            session['username'] = user.username
            user.update_last_login()
            flash(f'{user.username}님, 환영합니다!', 'success')
            return redirect(url_for('crud.index'))
        else:
            flash('사용자명 또는 비밀번호가 올바르지 않습니다.', 'error')
            return render_template('crud/login.html')
    
    return render_template('crud/login.html')

@auth.route('/logout')
def logout():
    session.clear()
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('auth.login'))

@auth.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('로그인이 필요합니다.', 'error')
        return redirect(url_for('auth.login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        flash('사용자 정보를 찾을 수 없습니다.', 'error')
        return redirect(url_for('auth.login'))
    
    # 최근 자세 분석 기록 가져오기 (최근 10개)
    recent_records = user.posture_records.order_by(PostureRecord.analysis_date.desc()).limit(10).all()
    
    # 통계 계산
    total_analyses = len(user.posture_records)
    if total_analyses > 0:
        avg_score = sum(record.calculate_overall_score() for record in user.posture_records) / total_analyses
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for record in user.posture_records:
            grade = record.calculate_overall_grade()
            grade_counts[grade] += 1
    else:
        avg_score = 0
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    return render_template('crud/profile.html', 
                         user=user, 
                         recent_records=recent_records,
                         total_analyses=total_analyses,
                         avg_score=avg_score,
                         grade_counts=grade_counts) 