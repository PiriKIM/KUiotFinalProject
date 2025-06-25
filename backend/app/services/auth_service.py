from sqlalchemy.orm import Session
from app.models import models
from app.models.schemas import UserCreate
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from typing import Optional

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class AuthService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_by_email(self, email: str):
        return self.db.query(models.User).filter(models.User.email == email).first()
    
    def get_user_by_username(self, username: str):
        return self.db.query(models.User).filter(models.User.username == username).first()
    
    def create_user(self, user_data: UserCreate):
        hashed_password = pwd_context.hash(user_data.password)
        db_user = models.User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            age=user_data.age,
            height=user_data.height,
            weight=user_data.weight
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    def authenticate_user(self, email: str, password: str):
        user = self.get_user_by_email(email)
        if not user or not pwd_context.verify(password, str(user.hashed_password)):
            return None
        return user
    
    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def get_current_user(self, token: str):
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: Optional[str] = payload.get("sub")
            if email is None:
                return None
        except JWTError:
            return None
        return self.get_user_by_email(email)
    
    def refresh_access_token(self, token: str):
        # TODO: 토큰 갱신 로직 구현
        return token 