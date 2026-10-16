# Relance RontoGPT proprement : ferme toute ancienne API (port 8760), puis
# ouvre le TUI avec le code ACTUEL. A utiliser apres chaque modification.
#   .\relancer.ps1
$port = if ($env:AURA_API_PORT) { [int]$env:AURA_API_PORT } else { 8760 }
$anciens = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($p in $anciens) {
    Write-Host "[relancer] arret de l'ancien processus $p (port $port)"
    Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
}
Set-Location $PSScriptRoot
# verifie le design (cadres de code + sources) avant d'ouvrir l'interface
$bun = Join-Path $PSScriptRoot "tui\node_modules\bun\bin\bun.exe"
if (Test-Path $bun) {
    Push-Location (Join-Path $PSScriptRoot "tui")
    & $bun run src/smoke_design.ts
    Pop-Location
}
uv run rontogpt
