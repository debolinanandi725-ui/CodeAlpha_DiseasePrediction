@echo off
cd /d %~dp0
call venv\Scripts\activate

echo ==========================
where python
echo ==========================

streamlit run app.py
pause