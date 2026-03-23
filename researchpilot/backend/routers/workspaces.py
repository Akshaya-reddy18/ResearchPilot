from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

from sqlalchemy.orm import Session

from db import get_session
from models.user import Workspace, Document, WorkspaceMember, User
from utils.deps import get_current_user, require_workspace_member


router = APIRouter(prefix="/api/v1", tags=["workspaces"])


class WorkspaceCreate(BaseModel):
  name: str
  description: Optional[str] = None


class WorkspaceOut(BaseModel):
  id: int
  name: str
  description: Optional[str] = None

  class Config:
    orm_mode = True


@router.get("/workspaces", response_model=List[WorkspaceOut])
def list_workspaces(
  db: Session = Depends(get_session),
  current_user: User = Depends(get_current_user),
):
  # workspaces owned by user
  owned = db.query(Workspace).filter(Workspace.owner_id == current_user.id)

  # workspaces where user is member
  member_ws_ids = (
    db.query(WorkspaceMember.workspace_id)
    .filter(WorkspaceMember.user_id == current_user.id)
    .subquery()
  )
  member = db.query(Workspace).filter(Workspace.id.in_(member_ws_ids))

  # union + distinct by id in Python
  all_ws: dict[int, Workspace] = {}
  for ws in list(owned) + list(member):
    all_ws[ws.id] = ws

  return list(all_ws.values())


@router.post("/workspaces", response_model=WorkspaceOut, status_code=status.HTTP_201_CREATED)
def create_workspace(
  payload: WorkspaceCreate,
  db: Session = Depends(get_session),
  current_user: User = Depends(get_current_user),
):
  ws = Workspace(name=payload.name, description=payload.description, owner_id=current_user.id)
  db.add(ws)
  db.commit()
  db.refresh(ws)

  # ensure owner is a member as well (optional but useful for queries)
  existing_member = (
    db.query(WorkspaceMember)
    .filter(WorkspaceMember.workspace_id == ws.id, WorkspaceMember.user_id == current_user.id)
    .first()
  )
  if not existing_member:
    db.add(WorkspaceMember(workspace_id=ws.id, user_id=current_user.id, role="owner"))
    db.commit()

  return ws


class DocumentCreate(BaseModel):
  workspace_id: int
  title: str


class DocumentOut(BaseModel):
  id: int
  title: str
  content: Optional[str] = None
  workspace_id: Optional[int] = None

  class Config:
    orm_mode = True


@router.get("/documents", response_model=List[DocumentOut])
def list_documents(
  workspace_id: int,
  db: Session = Depends(get_session),
  current_user: User = Depends(get_current_user),
):
  if not require_workspace_member(workspace_id, current_user, db):
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")

  docs = db.query(Document).filter(Document.workspace_id == workspace_id).all()
  return docs


@router.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def create_document(
  payload: DocumentCreate,
  db: Session = Depends(get_session),
  current_user: User = Depends(get_current_user),
):
  if not require_workspace_member(payload.workspace_id, current_user, db):
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")

  doc = Document(title=payload.title, content="", workspace_id=payload.workspace_id)
  db.add(doc)
  db.commit()
  db.refresh(doc)
  return doc


@router.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(
  doc_id: int,
  db: Session = Depends(get_session),
  current_user: User = Depends(get_current_user),
):
  doc = db.query(Document).filter(Document.id == doc_id).first()
  if not doc:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

  if doc.workspace_id and not require_workspace_member(doc.workspace_id, current_user, db):
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")

  return doc


class DocumentUpdate(BaseModel):
  title: Optional[str] = None
  content: Optional[str] = None


@router.put("/documents/{doc_id}", response_model=DocumentOut)
def update_document(
  doc_id: int,
  payload: DocumentUpdate,
  db: Session = Depends(get_session),
  current_user: User = Depends(get_current_user),
):
  doc = db.query(Document).filter(Document.id == doc_id).first()
  if not doc:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

  if doc.workspace_id and not require_workspace_member(doc.workspace_id, current_user, db):
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")

  if payload.title is not None:
    doc.title = payload.title
  if payload.content is not None:
    doc.content = payload.content

  db.add(doc)
  db.commit()
  db.refresh(doc)
  return doc


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
  doc_id: int,
  db: Session = Depends(get_session),
  current_user: User = Depends(get_current_user),
):
  doc = db.query(Document).filter(Document.id == doc_id).first()
  if not doc:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

  if doc.workspace_id and not require_workspace_member(doc.workspace_id, current_user, db):
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")

  db.delete(doc)
  db.commit()
  return
