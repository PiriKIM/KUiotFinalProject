from flask import Flask
from pathlib import Path
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, static_folder='apps/crud/static')  # 플라스크 인스턴스 생성

    app.config.from_mapping(
        SECRET_KEY='9dghh4g510frf7g1dgf2h6d4g',
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{Path(__file__).parent.parent / 'local.sqlite'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False
    )

    db.init_app(app)       # SQLAlchemy 연동
    Migrate(app, db)       # Migrate 연동

    from apps.crud import views as crud_views
    app.register_blueprint(crud_views.crud, url_prefix='/crud')  # Blueprint 등록

    return app
