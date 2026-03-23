from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Any
import json

from db import get_session
from utils.auth import get_user_by_token
from utils.deps import require_workspace_member
from models.user import Document, DocumentVersion

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        # workspace_id -> list of dict {"ws": WebSocket, "user_id": int, "email": str}
        self.active: Dict[str, List[Dict[str, Any]]] = {}

    async def connect(self, workspace_id: str, websocket: WebSocket, user_info: Dict[str, Any]):
        await websocket.accept()
        self.active.setdefault(workspace_id, []).append({"ws": websocket, **user_info})

    def disconnect(self, workspace_id: str, websocket: WebSocket):
        conns = self.active.get(workspace_id, [])
        self.active[workspace_id] = [c for c in conns if c["ws"] is not websocket]
        if not self.active[workspace_id]:
            self.active.pop(workspace_id, None)

    async def broadcast(self, workspace_id: str, message: dict, exclude: WebSocket | None = None):
        conns = self.active.get(workspace_id, [])
        for c in conns:
            ws = c["ws"]
            if exclude is not None and ws is exclude:
                continue
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                pass

    def list_users(self, workspace_id: str) -> List[Dict[str, Any]]:
        return [{"user_id": c.get("user_id"), "email": c.get("email")} for c in self.active.get(workspace_id, [])]


manager = ConnectionManager()


@router.websocket("/ws/workspace/{workspace_id}")
async def workspace_ws(websocket: WebSocket, workspace_id: str):
    # Expect token in query params: ?token=...
    token = websocket.query_params.get("token")
    dbgen = get_session()
    db = next(dbgen)
    try:
        user = get_user_by_token(token, db) if token else None
        if user is None:
            await websocket.close(code=1008)
            return

        # enforce workspace membership
        try:
            ws_id_int = int(workspace_id)
        except Exception:
            await websocket.close(code=1008)
            return
        if not require_workspace_member(ws_id_int, user, db):
            await websocket.close(code=1008)
            return

        user_info = {"user_id": user.id, "email": user.email}
        await manager.connect(workspace_id, websocket, user_info)

        # notify others of presence
        await manager.broadcast(workspace_id, {"type": "presence", "action": "join", "user": user_info}, exclude=websocket)

        # Send current presence list to connected client
        await websocket.send_text(json.dumps({"type": "presence_list", "users": manager.list_users(workspace_id)}))

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                continue

            mtype = msg.get("type")
            if mtype == "edit":
                # Broadcast edits to others
                payload = {"type": "edit", "user": user_info, "delta": msg.get("delta"), "doc_id": msg.get("doc_id")}
                await manager.broadcast(workspace_id, payload, exclude=websocket)

            elif mtype == "save":
                # Persist document content and create version
                doc_id = msg.get("doc_id")
                content = msg.get("content", "")
                if doc_id:
                    doc = db.query(Document).filter(Document.id == int(doc_id)).first()
                    if doc:
                        doc.content = content
                        from datetime import datetime
                        doc.updated_at = datetime.utcnow()
                        version = DocumentVersion(document_id=doc.id, content=content, author_id=user.id)
                        db.add(version)
                        db.commit()
                        db.refresh(doc)
                        await manager.broadcast(workspace_id, {"type": "saved", "doc_id": doc.id, "user": user_info})

            elif mtype == "cursor":
                payload = {"type": "cursor", "user": user_info, "cursor": msg.get("cursor")}
                await manager.broadcast(workspace_id, payload, exclude=websocket)

    except WebSocketDisconnect:
        manager.disconnect(workspace_id, websocket)
        await manager.broadcast(workspace_id, {"type": "presence", "action": "leave", "user": user_info})
    finally:
        try:
            db.close()
        except Exception:
            pass
