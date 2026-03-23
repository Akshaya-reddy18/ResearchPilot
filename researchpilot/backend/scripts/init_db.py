"""Initialize the database tables and optionally create an admin user.

Usage:
  python scripts/init_db.py

Optional env vars:
  ADMIN_EMAIL
  ADMIN_PASSWORD
"""
import os
from db import init_db, SessionLocal
from models.user import User
from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_admin(email: str, password: str):
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"Admin user {email} already exists")
            return
        admin = User(email=email, hashed_password=get_password_hash(password), full_name="Admin", is_superuser=True)
        db.add(admin)
        db.commit()
        print(f"Created admin user: {email}")
    finally:
        db.close()


if __name__ == "__main__":
    print("Initializing DB...")
    init_db()
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    if admin_email and admin_password:
        create_admin(admin_email, admin_password)
    else:
        print("No ADMIN_EMAIL/ADMIN_PASSWORD provided; skipping admin creation")
