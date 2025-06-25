from flask import Flask
from pathlib import Path
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)  ## 플라스크 인스턴스(객체) 생성


    app.config.from_mapping(SECRET_KEY = '9dghh4g510frf7g1dgf2h6d4g', 
                            SQLALCHEMY_DATABASE_URI = f"sqlite:///{Path(__file__).parent.parent / 'local.sqlite'}", 
                            SQLALCHEMY_TRACK_MODIFICATIONS = False)
    db.init_app(app)  ## SQLAlchemy와 app 연계
    Migrate(app, db)  ## Migrate와 app 연계


    from apps.crud import views as crud_views

    app.register_blueprint(crud_views.crud, url_prefix = '/crud')

    return app