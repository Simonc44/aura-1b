@echo off
rem Lance l'interface graphique Aura (pywebview) avec le Python du projet.
rem Double-cliquer sur ce fichier : pas besoin de taper de commande.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [Aura] Venv introuvable : lance d'abord "uv sync" dans ce dossier.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m aura.gui_web %*
