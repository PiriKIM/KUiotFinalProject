from datetime import datetime
from apps.app import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    last_login = db.Column(db.DateTime)
    
    # 관계 설정 - 사용자의 자세 분석 기록들
    posture_records = db.relationship('PostureRecord', backref='user', lazy=True, cascade='all, delete-orphan')
    
    @property
    def password(self):
        raise AttributeError("읽어 들일 수 없음")
    
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        self.last_login = datetime.now()
        db.session.commit()

class PostureRecord(db.Model):
    __tablename__ = "posture_record"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    analysis_date = db.Column(db.DateTime, default=datetime.now)
    
    # 자세 분석 결과 저장
    neck_angle = db.Column(db.Float)
    neck_grade = db.Column(db.String(1))  # A, B, C, D
    neck_description = db.Column(db.String(100))
    
    spine_is_hunched = db.Column(db.Boolean)
    spine_angle = db.Column(db.Float)
    
    shoulder_is_asymmetric = db.Column(db.Boolean)
    shoulder_height_difference = db.Column(db.Float)
    
    pelvic_is_tilted = db.Column(db.Boolean)
    pelvic_angle = db.Column(db.Float)
    
    spine_is_twisted = db.Column(db.Boolean)
    spine_alignment = db.Column(db.Float)
    
    # 종합 평가
    overall_score = db.Column(db.Integer)  # 0-100 점수
    overall_grade = db.Column(db.String(1))  # A, B, C, D
    
    def calculate_overall_score(self):
        """종합 점수 계산 (0-100)"""
        score = 100
        
        # 목 자세 점수 (30점)
        if self.neck_grade == 'A':
            score -= 0
        elif self.neck_grade == 'B':
            score -= 10
        elif self.neck_grade == 'C':
            score -= 20
        else:  # D
            score -= 30
        
        # 척추 곡률 점수 (25점)
        if self.spine_is_hunched:
            score -= 25
        
        # 어깨 비대칭 점수 (20점)
        if self.shoulder_is_asymmetric:
            score -= 20
        
        # 골반 기울기 점수 (15점)
        if self.pelvic_is_tilted:
            score -= 15
        
        # 척추 틀어짐 점수 (10점)
        if self.spine_is_twisted:
            score -= 10
        
        return max(0, score)
    
    def calculate_overall_grade(self):
        """종합 등급 계산"""
        score = self.calculate_overall_score()
        if score >= 90:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 50:
            return 'C'
        else:
            return 'D'