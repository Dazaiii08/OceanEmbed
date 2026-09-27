@echo off
title OceanEmbed Launcher

echo Starting OceanEmbed backend...
start "OceanEmbed Backend" cmd /k "cd /d "%~dp0backend" && python -m uvicorn main:app --reload"

echo Starting OceanEmbed frontend...
start "OceanEmbed Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo OceanEmbed is starting...
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo.

timeout /t 4 >nul
start http://localhost:5173