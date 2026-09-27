@echo off
title OceanEmbed Backend

cd /d "%~dp0backend"

echo Starting OceanEmbed FastAPI backend...
echo.

python -m uvicorn main:app --reload

pause