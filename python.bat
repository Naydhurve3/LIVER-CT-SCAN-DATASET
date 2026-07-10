@echo off
setlocal
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"

if exist "%PROJECT_PYTHON%" (
  "%PROJECT_PYTHON%" %*
  exit /b %ERRORLEVEL%
)

echo No project Python environment was found. 1>&2
echo Create one with: conda env create -f environment.yaml 1>&2
echo Then activate medsegx and run its python.exe directly, or create .venv here. 1>&2
exit /b 1
