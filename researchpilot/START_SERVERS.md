# How to Start the ResearchPilot Project

## Quick Start Instructions

### Option 1: Run in Separate Terminal Windows (Recommended)

#### Terminal 1 - Backend Server
```powershell
cd "D:\Research AI\ResearchHub-AI\researchpilot\backend"
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Wait for: `Application startup complete` and `Uvicorn running on http://127.0.0.1:8000`

#### Terminal 2 - Frontend Server
```powershell
cd "D:\Research AI\ResearchHub-AI\researchpilot\frontend"
npm run dev
```

Wait for: `Local: http://localhost:5173` (or similar port)

### Option 2: Run Both in PowerShell (Background)

#### Start Backend:
```powershell
cd "D:\Research AI\ResearchHub-AI\researchpilot\backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"
```

#### Start Frontend:
```powershell
cd "D:\Research AI\ResearchHub-AI\researchpilot\frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev"
```

## Access URLs

- **Backend API**: http://127.0.0.1:8000
- **Backend Docs**: http://127.0.0.1:8000/api/docs
- **Frontend**: http://localhost:5173 (check terminal for actual port)

## Troubleshooting

### Backend not starting?
1. Check if port 8000 is already in use: `netstat -ano | findstr ":8000"`
2. Make sure virtual environment is activated
3. Check for Python errors in the terminal

### Frontend not starting?
1. Make sure dependencies are installed: `npm install`
2. Check if port 5173 is available
3. Check for Node.js errors in the terminal

### Can't connect to localhost?
1. Make sure both servers are running
2. Check firewall settings
3. Try `127.0.0.1` instead of `localhost`
4. Check the actual port numbers in terminal output

