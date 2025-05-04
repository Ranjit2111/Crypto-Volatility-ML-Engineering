@echo off
REM Windows wrapper – activate venv, run full Phase‑2 pipeline, build plots
SETLOCAL
call "%~dp0venv\Scripts\activate.bat"
python scripts\fetch_data.py
python scripts\preprocess.py
python scripts\train.py
python scripts\predict.py
python scripts\generate_plots.py
call "%~dp0venv\Scripts\deactivate.bat"
ENDLOCAL