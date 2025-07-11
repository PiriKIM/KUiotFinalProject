from flask import Flask
from pathlib import Path
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    from apps.crud.models import User
    return User.query.get(int(user_id))

def create_app():
    app = Flask(__name__, static_folder='apps/crud/static')  # 플라스크 인스턴스 생성

    app.config.from_mapping(
        SECRET_KEY='9dghh4g510frf7g1dgf2h6d4g',
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{Path(__file__).parent.parent / 'local.sqlite'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False
    )

    db.init_app(app)       # SQLAlchemy 연동
    Migrate(app, db)       # Migrate 연동
    
    # Flask-Login 초기화
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = '로그인이 필요합니다.'
    login_manager.login_message_category = 'error'

    from apps.crud import views as crud_views
    from apps.crud import auth as auth_views
    from apps.crud import realtime_analysis as realtime_views
    
    app.register_blueprint(crud_views.crud, url_prefix='/crud')  # Blueprint 등록
    app.register_blueprint(auth_views.auth, url_prefix='/auth')  # 인증 Blueprint 등록
    app.register_blueprint(realtime_views.realtime_bp, url_prefix='/realtime')  # 실시간 분석 Blueprint 등록

    return app
