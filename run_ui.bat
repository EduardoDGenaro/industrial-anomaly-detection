@echo off
cd /d "%~dp0"
echo Iniciando o console de inspecao Streamlit...
call .venv\Scripts\activate.bat
python -m streamlit run ui/app.py
pause

