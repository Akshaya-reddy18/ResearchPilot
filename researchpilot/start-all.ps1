# Start Both Servers in Separate Windows
Write-Host "Starting ResearchPilot Servers..." -ForegroundColor Cyan
Write-Host "`nOpening Backend server window..." -ForegroundColor Yellow

# Start Backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$PSScriptRoot\start-backend.ps1`""

Start-Sleep -Seconds 2

Write-Host "Opening Frontend server window..." -ForegroundColor Yellow

# Start Frontend in new window
Start-Process powershell -ArgumentList "-NoExit", "-File", "`"$PSScriptRoot\start-frontend.ps1`""

Write-Host "`n✓ Both servers are starting in separate windows!" -ForegroundColor Green
Write-Host "`nAccess URLs:" -ForegroundColor Cyan
Write-Host "  Backend: http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  Backend Docs: http://127.0.0.1:8000/api/docs" -ForegroundColor White
Write-Host "  Frontend: Check the frontend terminal window for the URL" -ForegroundColor White
Write-Host "`nServers are running in separate windows!" -ForegroundColor Green
Write-Host "You can close this window." -ForegroundColor Gray

