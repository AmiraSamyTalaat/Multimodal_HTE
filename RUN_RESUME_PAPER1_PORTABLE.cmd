@echo off
setlocal
cd /d "%~dp0"
if not exist "postreview\source_snapshot\02_DATA\processed\ispy2_core_multimodal.npz" (
  echo First reconstruct the private GEO analysis matrix under postreview\source_snapshot\02_DATA\processed.
  exit /b 2
)
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  py -3 "postreview\code\run_resume.py" %*
) else (
  python "postreview\code\run_resume.py" %*
)
exit /b %ERRORLEVEL%
