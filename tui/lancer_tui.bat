@echo off
rem === RontoGPT TUI (OpenTUI) ===========================================
rem Demarre l'API locale (aura.api) si elle ne tourne pas deja, puis lance
rem l'interface terminal. Double-clic depuis ce dossier, ou depuis n'importe
rem ou (%~dp0 = dossier de ce .bat).
setlocal

set "TUI=%~dp0"
set "RACINE=%~dp0.."
rem bun installe par npm : le vrai binaire est sous node_modules\bun\bin\
rem (.bin\bun est un script sh, inutilisable par start/winpty)
set "BUN=%TUI%node_modules\bun\bin\bun.exe"
set "PY=%RACINE%\.venv\Scripts\python.exe"

if not exist "%BUN%" (
    echo [TUI] Bun n'est pas installe.
    echo [TUI] Ouvre un terminal dans tui\ et tape :  npm install
    pause
    exit /b 1
)
if not exist "%PY%" (
    echo [TUI] Cerveau Python absent : %PY%
    echo [TUI] Installe d'abord le projet avec scripts\installer.ps1
    pause
    exit /b 1
)

rem -- l'API tourne-t-elle deja ? (curl.exe est fourni par Windows 10+)
curl.exe -s -o nul --max-time 2 http://127.0.0.1:8760/v1/health
if not errorlevel 1 goto lance

echo [TUI] API absente : demarrage de aura.api...
start "RontoGPT API" /min "%PY%" -u -m aura.api
for /l %%i in (1,1,40) do (
    timeout /t 1 >nul
    curl.exe -s -o nul --max-time 2 http://127.0.0.1:8760/v1/health
    if not errorlevel 1 goto lance
)
echo [TUI] L'API n'a pas repondu en 40 s. Lance-la a la main pour voir l'erreur :
echo       cd /d "%RACINE%"
echo       .venv\Scripts\python.exe -m aura.api
pause
exit /b 1

:lance
cd /d "%RACINE%"
"%BUN%" "%TUI%src\index.ts"
if errorlevel 1 pause
endlocal
