from fastapi import APIRouter, Depends, HTTPException, Header, status
from typing import Optional
from pydantic import BaseModel
from db import get_session
from sqlalchemy.orm import Session
from models.user import Document, User
from utils.auth import decode_access_token
from utils.deps import get_current_user, require_workspace_member

router = APIRouter(prefix="/api/editor", tags=["editor"])


class DocumentIn(BaseModel):
    title: str
    content: Optional[str] = ""
    workspace_id: Optional[int] = None


class DocumentOut(BaseModel):
    id: int
    title: str
    content: Optional[str]
    workspace_id: Optional[int]


# Use dependency `get_current_user` from utils.deps to authenticate requests


@router.post("/doc", response_model=DocumentOut)
def create_doc(payload: DocumentIn, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if not payload.workspace_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="workspace_id is required")
    if not require_workspace_member(payload.workspace_id, user, session):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")
    doc = Document(title=payload.title, content=payload.content, workspace_id=payload.workspace_id)
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return DocumentOut(id=doc.id, title=doc.title, content=doc.content, workspace_id=doc.workspace_id)


@router.get("/doc/{doc_id}", response_model=DocumentOut)
def get_doc(doc_id: int, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    doc = session.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not require_workspace_member(doc.workspace_id, user, session):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")
    return DocumentOut(id=doc.id, title=doc.title, content=doc.content, workspace_id=doc.workspace_id)


@router.post("/doc/{doc_id}", response_model=DocumentOut)
def update_doc(doc_id: int, payload: DocumentIn, session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    doc = session.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not require_workspace_member(doc.workspace_id, user, session):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of workspace")
    doc.title = payload.title or doc.title
    doc.content = payload.content or doc.content
    doc.workspace_id = payload.workspace_id or doc.workspace_id
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return DocumentOut(id=doc.id, title=doc.title, content=doc.content, workspace_id=doc.workspace_id)
