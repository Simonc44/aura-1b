# =====================================================================
#  Aura — backend GPU (Vulkan) pour llama-server
#  Telecharge le build officiel win-vulkan-x64 de llama.cpp et le place
#  dans serveur/vulkan/ : serveur_lora le detecte alors tout seul et
#  passe en -ngl 99.
#
#  Mesure sur ce PC (scripts/bench_gpu.py, Llama-3.2-1B Q4_K_M, 256 tok) :
#      CPU    : decode 18,6-19,0 tok/s | prefill  66-73 tok/s
#      VULKAN : decode 20,4-20,6 tok/s | prefill 216-227 tok/s (x3,1)
#
#  Usage :
#    powershell -ExecutionPolicy Bypass -File scripts\installer_vulkan.ps1
#    powershell -File scripts\installer_vulkan.ps1 -Build b11111   # forcer
#    AURA_GPU=0  -> Aura ignore le build GPU (retour CPU garanti)
# =====================================================================
param(
    [string]$Build = "b11111",
    [string]$Destination = "",
    [switch]$Silencieux
)

$ErrorActionPreference = "Stop"
$RacineProjet = Split-Path -Parent $PSScriptRoot        # scripts/ -> racine
if (-not $Destination) { $Destination = Join-Path $RacineProjet "serveur\vulkan" }

$Tag = $Build
if ($Tag -notlike "b*") { $Tag = "b$Tag" }
$Nom = "llama-$Tag-bin-win-vulkan-x64.zip"
$Url = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/$Nom"
$Zip = Join-Path $env:TEMP $Nom

Write-Host "=== Backend GPU Vulkan pour Aura ===" -ForegroundColor Cyan
Write-Host "Release : $Tag"
Write-Host "Dest.   : $Destination"

# Pilote Vulkan deja present ? (vulkan-1.dll livre avec les pilotes GPU)
if (-not (Test-Path (Join-Path $env:SystemRoot "System32\vulkan-1.dll"))) {
    Write-Host "[X] Pilote Vulkan introuvable (System32\vulkan-1.dll)." -ForegroundColor Red
    Write-Host "    Mets a jour tes pilotes GPU (Intel/NVIDIA/AMD) puis relance."
    if (-not $Silencieux) { Read-Host "Entree pour fermer" }
    exit 1
}

if (-not (Test-Path $Zip)) {
    Write-Host "[..] Telechargement de $Nom (~35 Mo)..."
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
}

Write-Host "[..] Extraction..."
if (Test-Path $Destination) { Remove-Item -Recurse -Force $Destination }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Expand-Archive -Path $Zip -Destination $Destination -Force
Remove-Item $Zip -Force

$Exe = Join-Path $Destination "llama-server.exe"
if ((Test-Path $Exe) -and (Test-Path (Join-Path $Destination "ggml-vulkan.dll"))) {
    Write-Host "[OK] Backend GPU installe : $Exe" -ForegroundColor Green
    Write-Host "     Aura le choisit automatiquement (AURA_GPU=0 pour revenir au CPU)."
    # verification rapide du chargement du backend
    & $Exe --version 2>$null | Out-Null
    exit 0
}
Write-Host "[X] Extraction inattendue (voir $Destination)." -ForegroundColor Red
exit 1
