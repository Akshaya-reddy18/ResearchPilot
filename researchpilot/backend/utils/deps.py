from fastapi import Depends, HTTPException, status, Header
from typing import Optional
from db import get_session
from sqlalchemy.orm import Session
from models.user import User, WorkspaceMember
from utils.auth import decode_access_token


def get_db():
    return next(get_session())


def get_current_user(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)) -> User:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header")
    token = parts[1]
    payload = decode_access_token(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_workspace_member(workspace_id: int, user: User, db: Session) -> bool:
    # owner has implicit access
    ws = db.query(WorkspaceMember).filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user.id).first()
    if ws:
        return True
    # check workspace owner
    from models.user import Workspace
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if workspace and workspace.owner_id == user.id:
        return True
    return False
