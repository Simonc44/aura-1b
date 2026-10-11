"""Pont MCP Antigravity -> ile Coucou (Mochi)  (bati avec Freebuff).

Quoi
----
Un serveur **MCP** en stdio (JSON-RPC 2.0, une ligne par message), SANS
aucune dependance externe. Un agent (l'agent d'Antigravity, ou n'importe
quel client MCP) peut appeler l'outil ``notify_island`` pour faire parler
l'ile : c'est le pont entre Antigravity et Coucou.

Outils exposes
--------------
  notify_island(message, evenement?, agent?)  -> affiche un evenement dans l'ile
  island_status()                             -> l'ile est-elle joignable ?

Branchement dans Antigravity
----------------------------
Ajoute ceci dans ``~/.gemini/config/mcp_config.json`` (global) ou dans
``.agents/mcp_config.json`` (projet) ::

    {
      "mcpServers": {
        "coucou-island": {
          "command": "C:\\\\Users\\\\admin\\\\Desktop\\\\Dev\\\\IA\\\\aura-1b\\\\.venv\\\\Scripts\\\\python.exe",
          "args": ["C:\\\\Users\\\\admin\\\\Desktop\\\\Dev\\\\IA\\\\aura-1b\\\\scripts\\\\mcp_coucou.py"]
        }
      }
    }

Test manuel (l'echo d'une requete doit renvoyer du JSON) ::

    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts/mcp_coucou.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Le depot doit etre importable meme lance depuis n'importe quel dossier.
RACINE = Path(__file__).resolve().parent.parent
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from aura import coucou  # noqa: E402

PROTOCOLE = "2024-11-05"
NOM = "coucou-island"
VERSION = "1.0.0"

# Agent par defaut : une pastille distincte de la carte editeur "Antigravity".
AGENT_DEFAUT = "antigravity-agent"

OUTILS = [
    {
        "name": "notify_island",
        "description": (
            "Affiche un message dans l'ile Coucou (Mochi), en haut de l'ecran. "
            "A utiliser pour prevenir d'une fin de tache, d'une erreur, ou "
            "demander une information visible sans quitter l'editeur."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Texte court a afficher (une ligne).",
                },
                "evenement": {
                    "type": "string",
                    "enum": ["notification", "activite", "termine", "erreur"],
                    "description": "Style : notification (defaut), activite, termine, erreur.",
                },
                "agent": {
                    "type": "string",
                    "description": "Nom de la pastille (minuscules/chiffres/tirets).",
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "island_status",
        "description": "Verifie si l'ile Coucou est lancee et joignable (pipe local).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class _Erreur(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _traiter(methode: str, params: dict) -> dict:
    if methode == "initialize":
        return {
            "protocolVersion": PROTOCOLE,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": NOM, "version": VERSION},
        }
    if methode == "ping":
        return {}
    if methode == "tools/list":
        return {"tools": OUTILS}
    if methode == "tools/call":
        return _appeler(params.get("name") or "", params.get("arguments") or {})
    raise _Erreur(-32601, f"methode inconnue : {methode}")


def _appeler(nom: str, args: dict) -> dict:
    if nom == "island_status":
        dispo = coucou.est_disponible()
        txt = f"ile Coucou : {'joignable' if dispo else 'injoignable'} ({coucou.nom_pipe()})"
        return {"content": [{"type": "text", "text": txt}]}

    if nom == "notify_island":
        message = str(args.get("message") or "").strip()
        if not message:
            raise _Erreur(-32602, "le parametre 'message' est requis")
        try:
            resultat = coucou.notifier(
                message,
                evenement=str(args.get("evenement") or "notification"),
                agent=str(args.get("agent") or AGENT_DEFAUT),
            )
        except ValueError as e:
            raise _Erreur(-32602, str(e)) from e
        envoye = bool(resultat.get("envoye"))
        txt = ("envoye" if envoye else "non envoye") + f" : {resultat.get('raison')}"
        return {"content": [{"type": "text", "text": txt}], "isError": not envoye}

    raise _Erreur(-32602, f"outil inconnu : {nom}")


def _lire() -> dict | None:
    ligne = sys.stdin.buffer.readline()
    if not ligne:
        return None
    ligne = ligne.strip()
    if not ligne:
        return None
    return json.loads(ligne.decode("utf-8"))


def _ecrire(message: dict) -> None:
    donnees = json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n"
    sys.stdout.buffer.write(donnees)
    sys.stdout.buffer.flush()


def principal() -> int:
    while True:
        try:
            requete = _lire()
        except Exception:                      # ligne illisible : on continue
            continue
        if requete is None:                    # stdin ferme : l'agent est parti
            return 0
        id_requete = requete.get("id")
        if id_requete is None:                 # notification : pas de reponse
            continue
        try:
            resultat = _traiter(str(requete.get("method") or ""),
                                requete.get("params") or {})
            _ecrire({"jsonrpc": "2.0", "id": id_requete, "result": resultat})
        except _Erreur as e:
            _ecrire({"jsonrpc": "2.0", "id": id_requete,
                     "error": {"code": e.code, "message": e.message}})
        except Exception as e:                 # noqa: BLE001
            _ecrire({"jsonrpc": "2.0", "id": id_requete,
                     "error": {"code": -32603, "message": str(e)}})


if __name__ == "__main__":
    raise SystemExit(principal())
