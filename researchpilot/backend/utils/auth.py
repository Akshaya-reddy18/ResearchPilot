from jose import jwt, JWTError
from fastapi import HTTPException, status
from typing import Optional
import os

from db import get_session
from models.user import User

SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")


def get_user_by_token(token: str, db) -> Optional[User]:
    payload = decode_access_token(token)
    subject = payload.get("sub")
    if subject is None:
        return None
    user = db.query(User).filter(User.email == subject).first()
    return user
