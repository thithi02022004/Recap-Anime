@echo off
title Anime Recap Aligner
cd /d "%~dp0"

echo Dang khoi dong Anime Recap Aligner...
echo.
echo Moi ban truy cap: http://localhost:3000 neu trinh duyet khong tu mo.
echo.

start http://localhost:3000
npm run dev:all

pause
