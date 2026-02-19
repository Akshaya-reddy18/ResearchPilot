# Quick Start Guide

## 🚀 Run the Project

### Step 1: Start Backend Server

Open a **PowerShell terminal** and run:

```powershell
cd "D:\Research AI\ResearchHub-AI\researchpilot\backend"
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Wait for this message:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Step 2: Start Frontend Server

Open a **NEW PowerShell terminal** (keep backend running) and run:

```powershell
cd "D:\Research AI\ResearchHub-AI\researchpilot\frontend"
npm run dev
```

**Wait for this message:**
```
  ➜  Local:   http://localhost:5173/
```

### Step 3: Open in Browser

- **Backend API Docs**: http://127.0.0.1:8000/api/docs
- **Frontend App**: http://localhost:5173 (or the port shown in terminal)

## ✅ Verify Servers Are Running

### Check Backend:
Open: http://127.0.0.1:8000/health

Should return: `{"status":"healthy","service":"ResearchPilot AI Agent","message":"All systems operational"}`

### Check Frontend:
Open: http://localhost:5173

Should show the ResearchPilot landing page.

## 🔧 Troubleshooting

### Backend won't start?
1. Make sure virtual environment is activated
2. Check if port 8000 is in use: `netstat -ano | findstr ":8000"`
3. Look for Python errors in the terminal

### Frontend won't start?
1. Install dependencies: `npm install`
2. Check if Node.js is installed: `node --version`
3. Look for errors in the terminal

### Can't connect to localhost?
1. Make sure BOTH servers are running
2. Try `127.0.0.1` instead of `localhost`
3. Check firewall settings
4. Verify the ports in terminal output match what you're trying to access

## 📝 Notes

- Keep both terminal windows open while using the app
- Backend takes 10-30 seconds to start (loading AI models)
- Frontend may take a moment to compile on first run

