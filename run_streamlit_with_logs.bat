@echo off
REM Lance Streamlit et sauvegarde les logs dans un fichier (et en direct)

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ========================================
echo Lancement de Streamlit avec logs
echo ========================================
echo.

REM Creer un dossier logs s'il n'existe pas
if not exist "logs" mkdir logs

REM Obtenir la date/heure pour le nom du fichier
set timestamp=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%

set logfile=logs\streamlit_debug_%timestamp%.log

echo Les logs seront sauvegardes dans: %logfile%
echo Les logs defilent aussi dans la fenetre.
echo.
echo Appuie sur Ctrl+C dans ce terminal pour arreter Streamlit
echo.

REM Lancer Streamlit et dupliquer la sortie vers le fichier
powershell -NoProfile -Command "cmd /c ""streamlit run ui/app.py 2>&1"" | Tee-Object -FilePath '%logfile%'"

echo.
echo Streamlit arrete. Logs sauvegardes dans: %logfile%
pause
