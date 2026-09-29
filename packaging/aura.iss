; =====================================================================
;  Aura — installeur Windows (Inno Setup 6)
;  Construit par CI (release.yml) :
;    ISCC /DVersion=0.1.0 /DRacine=<racine du projet> packaging\aura.iss
;  Produit : dist\Aura-Setup-<version>.exe
;
;  Philosophie : l'installeur ne fait que COPIER les fichiers du projet
;  et creer les raccourcis ; la premiere execution (post-install) appelle
;  scripts\installer.ps1 qui construit le venv, installe les dependances
;  et telecharge le cerveau (807 Mo) — un seul et meme code d'installation
;  quel que soit le canal (Inno, PowerShell, git clone).
; =====================================================================

#ifndef Version
  #define Version "0.0.0"
#endif
#ifndef Racine
  #define Racine ".."
#endif

[Setup]
AppId={{A7E6A1C4-3B5E-4D2F-9C11-0AURA1B2C3D4E}
AppName=Aura
AppVersion={#Version}
AppPublisher=Simonc44
DefaultDirName={autopf}\Aura
DefaultGroupName=Aura
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=Aura-Setup-{#Version}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
; sans admin : installation utilisateur, {localappdata} pour le venv+GGUF
PrivilegesRequired=lowest
WizardStyle=modern

[Languages]
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"

[Files]
; sources + scripts (installer.ps1 s'attend a racine/scripts)
Source: "{#Racine}\aura\*"; DestDir: "{app}\aura"; Flags: recursesubdirs ignoreversion
Source: "{#Racine}\scripts\*"; DestDir: "{app}\scripts"; Flags: recursesubdirs ignoreversion
Source: "{#Racine}\packaging\*"; DestDir: "{app}\packaging"; Flags: recursesubdirs ignoreversion
Source: "{#Racine}\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Racine}\README.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; configuration centreee (profils rapide/equilibre/qualite)
Source: "{#Racine}\aura.toml"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; lanceur d'interface : %~dp0 = dossier du .bat, donc aucun chemin codé en dur
Source: "{#Racine}\packaging\Lancer Aura.bat"; DestDir: "{app}"; Flags: ignoreversion
; le .aef chiffre s'il existe deja (sinon telechargement ulterieur)
Source: "{#Racine}\aura_system.aef.enc"; DestDir: "{app}"; Flags: skipifsourcedoesntexist
Source: "{#Racine}\cle_publique.pub"; DestDir: "{app}"; Flags: skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\Aura"; Filename: "{app}\Lancer Aura.bat"
Name: "{autodesktop}\Aura"; Filename: "{app}\Lancer Aura.bat"
Name: "{autoprograms}\Desinstaurer Aura"; Filename: "{uninstallexe}"

[Run]
; premiere execution : venv + dependances + raccourci deja presents
Filename: "powershell.exe"; \
  Parameters: "-ExecutionPolicy Bypass -File ""{app}\scripts\installer.ps1"" -Silencieux -Destination ""{app}"""; \
  StatusMsg: "Installation du cerveau et des dependances (2-4 min)..."; \
  Flags: runascurrentuser postinstall skipifsilent

[UninstallDelete]
; venv, cache et modele restes sur le disque apres desinstallation
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\modeles"
Type: filesandordirs; Name: "{app}\.cache_aef"
Type: filesandordirs; Name: "{app}\serveur"
