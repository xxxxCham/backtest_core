@echo off
REM Lance Streamlit sans redirection (logs dans la fenetre)

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ========================================
echo Lancement de Streamlit (mode console)
echo ========================================
echo.
echo Appuie sur Ctrl+C dans ce terminal pour arreter Streamlit
echo.

streamlit run ui/app.py
