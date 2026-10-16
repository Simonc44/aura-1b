# =====================================================================
#  RontoGPT - Lanceur de l'interface terminal (OpenTUI)
#  Demarre l'API locale (aura/api.py) si elle ne tourne pas deja, puis
#  ouvre le TUI. Meme travail que lancer_tui.bat, en PowerShell.
#
#  Usage :
#    powershell -ExecutionPolicy Bypass -File lancer_tui.ps1
#    powershell -ExecutionPolicy Bypass -File lancer_tui.ps1 -Smoke
#    powershell -ExecutionPolicy Bypass -File lancer_tui.ps1 -SansApi
#
#  Options :
#    -Smoke     ne ouvre pas l'interface : joue les verifications
#               automatiques (client + ecran) et sort avec leur code
#    -SansApi   ne demarre pas l'API, dit seulement si elle repond
#    -ApiUrl    adresse du cerveau (defaut http://127.0.0.1:8760)
#
#  Si une API tournait DEJA avant ce script, elle date peut-etre d'une
#  version sans la route /v1/reset (commande /oubli) : dans ce cas,
#  ferme-la et relance.
# =====================================================================
[CmdletBinding()]
param(
    [switch]$Smoke,
    [switch]$SansApi,
    [string]$ApiUrl = 'http://127.0.0.1:8760',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Reste
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'     # pas de barre de progression reseau

$Tui    = $PSScriptRoot                         # tui/
$Racine = Split-Path -Parent $Tui               # racine du projet
$Bun    = Join-Path $Tui    'node_modules\bun\bin\bun.exe'
$Python = Join-Path $Racine '.venv\Scripts\python.exe'
$Index  = Join-Path $Tui    'src\index.ts'
$SmokeTs = Join-Path $Tui   'src\smoke.ts'

function Test-Api {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri ($Url + '/v1/health') -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

# ---------------------------------------------------------------
# 1. Les deux briques doivent etre la
# ---------------------------------------------------------------
if (-not (Test-Path $Bun)) {
    Write-Host "[X] Bun n'est pas installe pour l'interface terminal." -ForegroundColor Red
    Write-Host '    Ouvre un terminal dans tui\ et tape :  npm install'
    exit 1
}
if (-not (Test-Path $Python)) {
    Write-Host ('[X] Cerveau Python absent : ' + $Python) -ForegroundColor Red
    Write-Host '    Installe d''abord le projet : scripts\installer.ps1'
    exit 1
}

Write-Host 'RontoGPT - interface terminal' -ForegroundColor Cyan
Write-Host ('  interface : ' + $Tui)
Write-Host ('  cerveau   : ' + $ApiUrl)

# ---------------------------------------------------------------
# 2. L'API repond-elle ? Sinon on la demarre (sauf -SansApi)
# ---------------------------------------------------------------
$AideApi = '    A lancer a la main :  cd ' + $Racine + ' ; .venv\Scripts\python.exe -m aura.api'

if (Test-Api -Url $ApiUrl) {
    Write-Host '[OK] API deja en ligne.' -ForegroundColor Green
} elseif ($SansApi) {
    Write-Host ('[X] API injoignable sur ' + $ApiUrl + ' (-SansApi : je ne la demarre pas).') -ForegroundColor Red
    Write-Host $AideApi
    exit 1
} else {
    Write-Host '[..] API absente : demarrage de aura.api...' -ForegroundColor Yellow
    $proc = Start-Process -FilePath $Python -ArgumentList '-u', '-m', 'aura.api' `
                          -WorkingDirectory $Racine -WindowStyle Minimized -PassThru

    $pret = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 1
        if ($proc.HasExited) {
            Write-Host ('[X] L''API s''est arretee aussitot (code ' + $proc.ExitCode + ').') -ForegroundColor Red
            Write-Host $AideApi
            exit 1
        }
        if (Test-Api -Url $ApiUrl) { $pret = $true; break }
    }
    if (-not $pret) {
        Write-Host '[X] Pas de reponse de l''API en 40 s.' -ForegroundColor Red
        Write-Host $AideApi
        exit 1
    }
    Write-Host ('[OK] API prete (PID ' + $proc.Id + ').') -ForegroundColor Green
}

# ---------------------------------------------------------------
# 3. Verification automatique, ou interface interactive
# ---------------------------------------------------------------
if ($Smoke) {
    Write-Host ''
    Write-Host '=== Client (src\smoke.ts) ===' -ForegroundColor Cyan
    & $Bun $SmokeTs
    $codeClient = $LASTEXITCODE

    Write-Host ''
    Write-Host '=== Ecran complet (src\index.ts --smoke) ===' -ForegroundColor Cyan
    & $Bun $Index --smoke
    $codeEcran = $LASTEXITCODE

    Write-Host ''
    if ($codeClient -eq 0 -and $codeEcran -eq 0) {
        Write-Host '[OK] Verifications passees.' -ForegroundColor Green
        exit 0
    }
    Write-Host ('[X] Echec : client=' + $codeClient + ' ecran=' + $codeEcran) -ForegroundColor Red
    exit 1
}

Write-Host '[OK] Ouverture de l''interface (Ctrl+C ou /quitter pour sortir).' -ForegroundColor Green
Write-Host ''
& $Bun $Index @Reste
exit $LASTEXITCODE
