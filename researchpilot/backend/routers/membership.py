from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from db import get_session
from sqlalchemy.orm import Session
from typing import List
from models.user import WorkspaceMember, User, Workspace
from utils.deps import get_current_user, require_workspace_member

router = APIRouter(prefix="/api/workspaces", tags=["membership"])


class AddMemberIn(BaseModel):
    email: str
    role: str = "member"


class MemberOut(BaseModel):
    user_id: int
    email: str
    role: str


@router.get("/{workspace_id}/members", response_model=List[MemberOut])
def list_members(workspace_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    if not require_workspace_member(workspace_id, current_user, session):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")
    members = session.query(WorkspaceMember).filter(WorkspaceMember.workspace_id == workspace_id).all()
    out = []
    for m in members:
        user = session.query(User).filter(User.id == m.user_id).first()
        if user:
            out.append(MemberOut(user_id=user.id, email=user.email, role=m.role))
    return out


@router.post("/{workspace_id}/members", response_model=MemberOut)
def add_member(workspace_id: int, payload: AddMemberIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    # only workspace owner may add members
    workspace = session.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if workspace.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only workspace owner can add members")

    user = session.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    existing = session.query(WorkspaceMember).filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already a member")

    member = WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role=payload.role)
    session.add(member)
    session.commit()
    return MemberOut(user_id=user.id, email=user.email, role=payload.role)


@router.delete("/{workspace_id}/members/{user_id}")
def remove_member(workspace_id: int, user_id: int, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    workspace = session.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    if workspace.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only workspace owner can remove members")

    member = session.query(WorkspaceMember).filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    session.delete(member)
    session.commit()
    return {"status": "removed"}
