Integration testing instructions

Requirements:
- Backend server running at http://127.0.0.1:8000
- Python venv with `requests` installed

Run from project `backend` folder:

```
.\venv\Scripts\python.exe -m pip install requests
.\venv\Scripts\python.exe tests\integration_test.py
```

The script will register two test users, create a workspace (direct DB insert), create a document, export DOCX, add/remove a member, and print results.
