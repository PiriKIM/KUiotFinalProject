from flask import Flask
from pathlib import Path
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    # 현재 파일의 위치를 기준으로 static 폴더 경로 설정
    static_folder_path = Path(__file__).parent / 'crud' / 'static'
    app = Flask(__name__, static_folder=str(static_folder_path), static_url_path='/static')

    app.config.from_mapping(
        SECRET_KEY='9dghh4g510frf7g1dgf2h6d4g',
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{Path(__file__).parent.parent / 'local.sqlite'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False
    )

    db.init_app(app)       # SQLAlchemy 연동
    Migrate(app, db)       # Migrate 연동

    from apps.crud import views as crud_views
    from apps.crud import auth as auth_views
    
    app.register_blueprint(crud_views.crud, url_prefix='/crud')  # Blueprint 등록
    app.register_blueprint(auth_views.auth, url_prefix='/auth')  # 인증 Blueprint 등록

    return app
