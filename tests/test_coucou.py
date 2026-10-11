"""Pont RontoGPT -> ile Coucou (named pipe) + route /v1/notify.

Aucun Coucou n'est requis : on teste la construction du message, la
validation, le repli Windows/hors-Windows, et la route HTTP en simulant
l'ecriture sur le pipe.
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from aura import coucou


# ── construction du message ────────────────────────────────────────────

class TestPayload:
    def test_evenements_mappes(self):
        assert coucou._payload("salut", "activite")["hook_event_name"] == "UserPromptSubmit"
        assert coucou._payload("fini", "termine")["hook_event_name"] == "Stop"
        assert coucou._payload("boom", "erreur")["hook_event_name"] == "StopFailure"
        assert coucou._payload("info", "notification")["hook_event_name"] == "Notification"

    def test_message_et_agent_inclus(self):
        corps = coucou._payload("  reponse prete  ", "termine", "aura")
        assert corps["message"] == "reponse prete"
        assert corps["coucou_agent"] == "aura"

    def test_message_vide_absent(self):
        assert "message" not in coucou._payload("", "termine")

    def test_evenement_inconnu_leve(self):
        with pytest.raises(ValueError):
            coucou._payload("x", "nimportequoi")

    @pytest.mark.parametrize("mauvais", ["Claude", "claude", "Avec Espace",
                                         "Majuscule", "au_des", "a" * 25, "", None])
    def test_agent_invalide_ignore(self, mauvais):
        assert "coucou_agent" not in coucou._payload("x", "termine", mauvais)

    def test_nom_pipe(self):
        assert coucou._nom_pipe_pour("S-1-5-21-42") == r"\\.\pipe\coucou-S-1-5-21-42"
        assert coucou._nom_pipe_pour("") == r"\\.\pipe\coucou-user"


# ── notifier : repli et coutures ───────────────────────────────────────

class TestNotifier:
    def test_interrupteur_coupe_tout(self, monkeypatch):
        monkeypatch.setenv("AURA_COUCOU", "0")
        r = coucou.notifier("bonjour")
        assert r["envoye"] is False and "desactive" in r["raison"]

    def test_succes_simule(self, monkeypatch):
        monkeypatch.delenv("AURA_COUCOU", raising=False)
        monkeypatch.setattr(coucou, "_sur_windows", lambda: True)
        monkeypatch.setattr(coucou, "nom_pipe", lambda: r"\\.\pipe\coucou-test")
        vus = {}

        def faux(nom, ligne, timeout):
            vus["nom"], vus["ligne"] = nom, ligne
            return True, None

        monkeypatch.setattr(coucou, "_envoyer", faux)
        r = coucou.termine("reponse prete")
        assert r["envoye"] is True and r["evenement"] == "Stop"
        # la ligne ecrite est bien du JSON termine par un unique '\n'
        assert vus["ligne"].endswith("\n") and not vus["ligne"].endswith("\r\n")
        assert json.loads(vus["ligne"])["message"] == "reponse prete"

    def test_echec_pipe_absent(self, monkeypatch):
        monkeypatch.delenv("AURA_COUCOU", raising=False)
        monkeypatch.setattr(coucou, "_sur_windows", lambda: True)
        monkeypatch.setattr(coucou, "nom_pipe", lambda: r"\\.\pipe\coucou-test")
        monkeypatch.setattr(coucou, "_envoyer", lambda *a: (False, "fichier introuvable"))
        r = coucou.notifier("coucou ?")
        assert r["envoye"] is False and "introuvable" in r["raison"]

    def test_hors_windows(self, monkeypatch):
        monkeypatch.delenv("AURA_COUCOU", raising=False)
        monkeypatch.setattr(coucou, "_sur_windows", lambda: False)
        r = coucou.notifier("bonjour")
        assert r["envoye"] is False and "Windows" in r["raison"]

    def test_disponible_faux_si_coupe(self, monkeypatch):
        monkeypatch.setenv("AURA_COUCOU", "0")
        assert coucou.est_disponible() is False


# ── ligne de commande (pour un agent d'IDE, un script, un hook) ─────────

class TestCli:
    def test_message_requis(self):
        with pytest.raises(SystemExit) as e:
            coucou.principal([])
        assert e.value.code == 2

    def test_evenement_inconnu(self):
        with pytest.raises(SystemExit) as e:
            coucou.principal(["x", "--evenement", "blabla"])
        assert e.value.code == 2

    def test_json_et_code_succes(self, monkeypatch, capsys):
        monkeypatch.setattr(
            coucou, "notifier", lambda *a, **k: {"envoye": True, "raison": "envoye"})
        assert coucou.principal(["salut", "--json"]) == 0
        assert '"envoye": true' in capsys.readouterr().out

    def test_non_envoye_code_1(self, monkeypatch):
        monkeypatch.setattr(
            coucou, "notifier", lambda *a, **k: {"envoye": False, "raison": "absent"})
        assert coucou.principal(["salut"]) == 1


# ── route HTTP /v1/notify + /v1/coucou ─────────────────────────────────

@pytest.fixture()
def serveur():
    from http.server import ThreadingHTTPServer
    from aura import api as api_mod
    etat = api_mod._Etat(smoke=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), api_mod._creer_handler(etat))
    fil = threading.Thread(target=srv.serve_forever, daemon=True)
    fil.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(base, route):
    try:
        with urllib.request.urlopen(base + route, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def _post(base, route, corps):
    req = urllib.request.Request(
        base + route, data=json.dumps(corps).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


class TestRoute:
    def test_coucou_disponible_expose(self, serveur):
        code, rep = _get(serveur, "/v1/coucou")
        assert code == 200 and rep["ok"] is True
        assert isinstance(rep["disponible"], bool)
        assert "termine" in rep["evenements"]

    def test_notify_message_vide_400(self, serveur):
        code, rep = _post(serveur, "/v1/notify", {"message": "  "})
        assert code == 400 and rep["ok"] is False

    def test_notify_transmet_a_coucou(self, serveur, monkeypatch):
        from aura import coucou as coucou_mod
        vus = {}

        def faux(message, *, evenement="notification", agent="aura", timeout=0.6):
            vus.update(message=message, evenement=evenement, agent=agent)
            return {"envoye": True, "evenement": "Stop", "raison": "envoye"}

        monkeypatch.setattr(coucou_mod, "notifier", faux)
        code, rep = _post(serveur, "/v1/notify",
                          {"message": "c'est pret", "evenement": "termine",
                           "agent": "aura"})
        assert code == 200 and rep["ok"] is True and rep["envoye"] is True
        assert vus == {"message": "c'est pret", "evenement": "termine", "agent": "aura"}

    def test_notify_evenement_inconnu_400(self, serveur):
        code, rep = _post(serveur, "/v1/notify",
                          {"message": "x", "evenement": "blabla"})
        assert code == 400 and rep["ok"] is False
