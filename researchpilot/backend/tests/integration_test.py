import time
import requests
from db import SessionLocal
from models.user import Workspace

BASE = "http://127.0.0.1:8000"


def register(email, password):
    r = requests.post(f"{BASE}/api/auth/register", json={"email": email, "password": password})
    try:
        r.raise_for_status()
    except Exception:
        print('register failed:', r.status_code, r.text)
        return None
    return r.json()


def token(email, password):
    r = requests.post(f"{BASE}/api/auth/token", json={"email": email, "password": password})
    try:
        r.raise_for_status()
    except Exception:
        print('token failed:', r.status_code, r.text)
        return None
    return r.json()["access_token"]


def create_workspace_db(name, owner_id):
    db = SessionLocal()
    try:
        ws = Workspace(name=name, owner_id=owner_id)
        db.add(ws)
        db.commit()
        db.refresh(ws)
        return ws.id
    finally:
        db.close()


def main():
    # simple integration flow
    e1 = f"test1+{int(time.time())}@example.com"
    p1 = "password123"
    print("Registering user1...", e1)
    u1 = register(e1, p1)
    tok1 = token(e1, p1)
    headers = {"Authorization": f"Bearer {tok1}"}

    print("Creating workspace in DB for user1")
    ws_id = create_workspace_db("Test Workspace", u1["id"])
    print("Workspace id:", ws_id)

    print("Creating document via API")
    r = requests.post(f"{BASE}/api/editor/doc", json={"title": "Test Doc", "content": "Hello world", "workspace_id": ws_id}, headers=headers)
    r.raise_for_status()
    doc = r.json()
    print("Created doc id", doc["id"]) 

    print("Exporting DOCX")
    r = requests.get(f"{BASE}/api/export/doc/{doc['id']}/docx", headers=headers)
    if r.status_code == 200:
        print("DOCX export OK, size=", len(r.content))
    else:
        print("DOCX export failed", r.status_code, r.text)

    print("Listing members")
    r = requests.get(f"{BASE}/api/workspaces/{ws_id}/members", headers=headers)
    r.raise_for_status()
    print("Members:", r.json())

    print("Registering user2 and adding as member")
    e2 = f"test2+{int(time.time())}@example.com"
    p2 = "password123"
    u2 = register(e2, p2)
    r = requests.post(f"{BASE}/api/workspaces/{ws_id}/members", json={"email": e2, "role": "member"}, headers=headers)
    r.raise_for_status()
    print("Added member:", r.json())

    print("Members after add:")
    r = requests.get(f"{BASE}/api/workspaces/{ws_id}/members", headers=headers)
    r.raise_for_status()
    print(r.json())

    print("Removing member")
    r = requests.delete(f"{BASE}/api/workspaces/{ws_id}/members/{u2['id']}", headers=headers)
    r.raise_for_status()
    print("Removed")

    print("Integration test completed successfully")


if __name__ == "__main__":
    main()
