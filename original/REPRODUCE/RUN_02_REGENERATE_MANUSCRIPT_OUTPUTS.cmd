@echo off
cd /d "%~dp0.."
python "%~dp0MAKE_FIGURE1_WORKFLOW.py"
python "%~dp0REGENERATE_MANUSCRIPT_OUTPUTS.py"
pause
