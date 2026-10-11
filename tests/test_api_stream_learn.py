"""Streaming SSE (/v1/ask/stream), edition de faits (/v1/learn) et pont MCP.

Aucun cerveau reel : on injecte une fausse IA qui publie des morceaux, et
on simule l'ecriture des faits/corrections. Le serveur MCP est teste par
ses fonctions pures (_traiter / _appeler).
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from aura import api as api_mod
from aura import llama_cerveau


# ── serveur HTTP de test ───────────────────────────────────────────────

@pytest.fixture()
def serveur():
    from http.server import ThreadingHTTPServer
    etat = api_mod._Etat(smoke=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), api_mod._creer_handler(etat))
    fil = threading.Thread(target=srv.serve_forever, daemon=True)
    fil.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", etat
    srv.shutdown()


def _post(base, route, corps):
    req = urllib.request.Request(
        base + route, data=json.dumps(corps).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


class _FauxIA:
    """Publie des morceaux comme le ferait le vrai cerveau."""

    def __init__(self, morceaux):
        self.morceaux = morceaux

    def executer_detaille(self, question):          # noqa: ARG002
        for m in self.morceaux:
            llama_cerveau.publier_flux(m)
        return {"reponse": "".join(self.morceaux), "experts": {"test"},
                "cerveau_choisi": "faux"}


# ── streaming ──────────────────────────────────────────────────────────

class TestStream:
    def test_sans_cerveau_503(self, serveur):
        base, _ = serveur
        code, corps = _post(base, "/v1/ask/stream", {"question": "bonjour"})
        assert code == 503

    def test_question_vide_400(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA(["x"])
        code, _ = _post(base, "/v1/ask/stream", {"question": "  "})
        assert code == 400

    def test_flux_des_morceaux_puis_final(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA(["Bon", "jour"])
        code, corps = _post(base, "/v1/ask/stream", {"question": "salut"})
        assert code == 200
        assert '"type": "morceau"' in corps
        assert '"texte": "Bon"' in corps and '"texte": "jour"' in corps
        assert '"type": "final"' in corps
        assert '"reponse": "Bonjour"' in corps
        assert "data: [DONE]" in corps

    def test_verrou_libere_apres_le_flux(self, serveur):
        base, etat = serveur
        etat.ia = _FauxIA(["a"])
        _post(base, "/v1/ask/stream", {"question": "1"})
        # le verrou du flux doit etre rendu pour la requete suivante
        code, _ = _post(base, "/v1/ask/stream", {"question": "2"})
        assert code == 200


# ── edition de faits (/v1/learn) ───────────────────────────────────────

class TestLearn:
    def test_fait_ajoute(self, serveur, monkeypatch):
        base, _ = serveur
        from aura import graphe_faits
        vus = {}
        monkeypatch.setattr(graphe_faits, "ajouter",
                            lambda s, r, o, src="manuel": vus.update(
                                s=s, r=r, o=o, src=src) or True)
        code, corps = _post(base, "/v1/learn",
                            {"sujet": "Paris", "relation": "est la capitale de",
                             "objet": "la France", "source": "test"})
        assert code == 200
        assert '"type": "fait"' in corps and '"ajoute": true' in corps
        assert vus == {"s": "Paris", "r": "est la capitale de",
                       "o": "la France", "src": "test"}

    def test_correction_enregistree(self, serveur, monkeypatch):
        base, _ = serveur
        from aura import autoamelioration
        vus = {}
        monkeypatch.setattr(autoamelioration, "enregistrer_correction",
                            lambda *a, **k: vus.update(args=a))
        code, corps = _post(base, "/v1/learn",
                            {"question": "carre de 5", "correcte": "25"})
        assert code == 200 and '"type": "correction"' in corps
        assert vus["args"][0] == "carre de 5" and vus["args"][2] == "25"

    def test_champs_manquants_400(self, serveur):
        base, _ = serveur
        code, _ = _post(base, "/v1/learn", {"sujet": "x"})
        assert code == 400


# ── serveur MCP (pont Antigravity -> Coucou) ───────────────────────────

class TestMcp:
    @pytest.fixture(autouse=True)
    def _module(self):
        import scripts.mcp_coucou as mcp
        self.mcp = mcp

    def test_initialize(self):
        r = self.mcp._traiter("initialize", {})
        assert r["protocolVersion"] == self.mcp.PROTOCOLE
        assert r["serverInfo"]["name"] == "coucou-island"

    def test_tools_list(self):
        noms = {t["name"] for t in self.mcp._traiter("tools/list", {})["tools"]}
        assert noms == {"notify_island", "island_status"}

    def test_notify_appelle_le_pont(self, monkeypatch):
        from aura import coucou
        vus = {}
        monkeypatch.setattr(coucou, "notifier", lambda message, **k: (
            vus.update(message=message, **k)
            or {"envoye": True, "raison": "envoye"}))
        r = self.mcp._traiter("tools/call", {
            "name": "notify_island",
            "arguments": {"message": "c fini", "evenement": "termine"}})
        assert r["isError"] is False
        assert vus["message"] == "c fini" and vus["evenement"] == "termine"

    def test_notify_message_requis(self):
        with pytest.raises(self.mcp._Erreur):
            self.mcp._traiter("tools/call", {"name": "notify_island",
                                             "arguments": {}})

    def test_methode_inconnue(self):
        with pytest.raises(self.mcp._Erreur):
            self.mcp._traiter("methode.bidon", {})
