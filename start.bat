@echo off
rem Start the diabetes risk check (API + web UI) on this PC. Double-click, or run: start.bat [--no-browser] [--check]
rem Uses the da-ml conda environment (see README.md, D-59).
cd /d "%~dp0"

set "PY=%USERPROFILE%\miniconda3\envs\da-ml\python.exe"
if not exist "%PY%" set "PY=%USERPROFILE%\anaconda3\envs\da-ml\python.exe"
if not exist "%PY%" (
    echo Could not find the da-ml environment.
    echo Create it first with:  conda env create -f environment.yml
    pause
    exit /b 1
)

"%PY%" run.py %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" pause
exit /b %CODE%
