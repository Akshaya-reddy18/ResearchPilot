# from fastapi import APIRouter, Depends, HTTPException, status
# from pydantic import BaseModel, EmailStr
# from passlib.context import CryptContext
# from datetime import datetime, timedelta
# from jose import JWTError, jwt
# import os
# from pydantic import field_validator

# from db import get_session, init_db
# from models.user import User
# from sqlalchemy.orm import Session

# router = APIRouter(prefix="/api/auth", tags=["auth"])

# # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# pwd_context = CryptContext(
#     schemes=["bcrypt"],
#     deprecated="auto",
#     bcrypt__truncate_error=False
# )

# SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


# class Token(BaseModel):
#     access_token: str
#     token_type: str = "bearer"


# class TokenData(BaseModel):
#     email: EmailStr | None = None


# class UserCreate(BaseModel):
#     email: EmailStr
#     password: str
#     full_name: str | None = None

#     @field_validator("password", mode="before")
#     @classmethod
#     def validate_password(cls, value: str) -> str:
#         if not isinstance(value, str):
#             raise ValueError("Password must be a string")

#         value = value.strip()

#         if len(value) < 8:
#             raise ValueError("Password must be at least 8 characters")
#         if len(value) > 72:
#             raise ValueError("Password must be at most 72 characters")

#         return value


# class UserOut(BaseModel):
#     id: int
#     email: EmailStr
#     full_name: str | None = None


# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)


# # def get_password_hash(password):
# #     return pwd_context.hash(password)
# def get_password_hash(password: str):
#     return pwd_context.hash(password.strip())


# def create_access_token(data: dict, expires_delta: timedelta | None = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt


# @router.on_event("startup")
# def startup_event():
#     # Ensure DB tables exist on startup
#     try:
#         init_db()
#     except Exception:
#         pass


# @router.post("/register", response_model=UserOut)
# def register(user: UserCreate, session: Session = Depends(get_session)):
#     try:
#         import logging
#         logging.info(f"Register called for email={user.email!r}, password_len={len(user.password) if user.password is not None else 'None'}")
#         existing = session.query(User).filter(User.email == user.email).first()
#         if existing:
#             raise HTTPException(status_code=400, detail="Email already registered")
#         db_user = User(email=user.email, hashed_password=get_password_hash(user.password), full_name=user.full_name)
#         session.add(db_user)
#         session.commit()
#         session.refresh(db_user)
#         return UserOut(id=db_user.id, email=db_user.email, full_name=db_user.full_name)
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback, logging
#         logging.exception("Error in register")
#         raise HTTPException(status_code=500, detail=str(e))


# @router.post("/token", response_model=Token)
# def login_for_access_token(form_data: UserCreate, session: Session = Depends(get_session)):
#     user = session.query(User).filter(User.email == form_data.email).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
#     access_token = create_access_token(data={"sub": user.email})
#     return Token(access_token=access_token)
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
import os

from db import get_session, init_db
from models.user import User
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Clean bcrypt config (no hacks needed now)
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


# =========================
# SCHEMAS
# =========================

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: EmailStr | None = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Password must be a string")

        value = value.strip()

        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(value) > 72:
            raise ValueError("Password must be at most 72 characters")

        return value


# 🔥 NEW: Separate login schema (important fix)
class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str | None = None


# =========================
# UTILS
# =========================

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str):
    password = password.strip()[:72]
    return pwd_context.hash(password.strip())


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# =========================
# STARTUP
# =========================

@router.on_event("startup")
def startup_event():
    try:
        init_db()
    except Exception:
        pass


# =========================
# ROUTES
# =========================

@router.post("/register", response_model=UserOut)
def register(user: UserCreate, session: Session = Depends(get_session)):
    try:
        existing = session.query(User).filter(User.email == user.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        db_user = User(
            email=user.email,
            hashed_password=get_password_hash(user.password),
            full_name=user.full_name
        )

        session.add(db_user)
        session.commit()
        session.refresh(db_user)

        return UserOut(
            id=db_user.id,
            email=db_user.email,
            full_name=db_user.full_name
        )

    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.exception("Error in register")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/token", response_model=Token)
def login_for_access_token(form_data: UserLogin, session: Session = Depends(get_session)):
    user = session.query(User).filter(User.email == form_data.email).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Store email in `sub` so utils.auth and deps.get_current_user can
    # reliably look up the user by email from the token payload.
    access_token = create_access_token(data={"sub": user.email})

    return Token(access_token=access_token)