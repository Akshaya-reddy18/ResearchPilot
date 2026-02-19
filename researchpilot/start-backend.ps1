# Start Backend Server
Write-Host "Starting Backend Server..." -ForegroundColor Cyan
cd "D:\Research AI\ResearchHub-AI\researchpilot\backend"
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

