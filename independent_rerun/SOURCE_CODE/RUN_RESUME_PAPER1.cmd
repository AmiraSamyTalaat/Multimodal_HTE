@echo off
setlocal
set "KAGGLE_STAGE_ROOT=%~dp0"
py -3.14 "%KAGGLE_STAGE_ROOT%resume_paper1.py" %*
exit /b %ERRORLEVEL%
