"""Configuration centralisee, API locale /v1/* et verification de mises a jour.

Ces trois briques sont celles de l'industrialisation : un fichier de
config unique, un coeur expose en JSON pour tous les front-ends, et une
detection de release. Tout se teste SANS modele (la CI n'a ni GGUF ni GPU).
"""
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from aura import config, maj


@pytest.fixture(autouse=True)
def _environnement_isole():
    """config.appliquer() pose des AURA_* dans l'ENVIRONNEMENT DU PROCESS :
    c'est voulu en production, mais entre deux tests ca pollue toute la
    suite (un AURA_AGENTS=0 rendrait les tests d'agents silencieusement
    faux). On snapshot/restaure tout, test par test."""
    avant = dict(os.environ)
    yield
    for k in set(os.environ) - set(avant):
        os.environ.pop(k, None)
    for k, v in avant.items():
        os.environ[k] = v


# ── configuration aura.toml + profils ──────────────────────────────────

class TestConfig:
    def test_fichier_absent_ne_casse_rien(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AURA_PROFIL", raising=False)
        pose = config.appliquer(chemin=tmp_path / "inexistant.toml")
        assert pose["AURA_PROFIL"] == "equilibre"

    def test_profil_rapide_pose_ses_variables(self, tmp_path, monkeypatch):
        for k in ("AURA_AGENTS", "AURA_VERIF_WEB", "AURA_BUDGET_S"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv("AURA_PROFIL", raising=False)
        pose = config.appliquer(chemin=tmp_path / "x.toml", profil="rapide")
        assert pose["AURA_PROFIL"] == "rapide"
        assert os.environ["AURA_AGENTS"] == "0"
        assert os.environ["AURA_BUDGET_S"] == "15"

    def test_variable_utilisateur_ne_s_est_pas_ecrasee(self, tmp_path, monkeypatch):
        # priorite 1 : l'environnement de l'utilisateur gagne sur le fichier
        monkeypatch.setenv("AURA_BUDGET_S", "999")
        (tmp_path / "aura.toml").write_text(
            "[qualite]\nbudget_s = 42\n", encoding="utf-8")
        config.appliquer(chemin=tmp_path / "aura.toml")
        assert os.environ["AURA_BUDGET_S"] == "999"

    def test_bool_toml_devient_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AURA_SERVEUR", raising=False)
        (tmp_path / "aura.toml").write_text(
            "[serveur]\nactif = true\n", encoding="utf-8")
        config.appliquer(chemin=tmp_path / "aura.toml")
        assert os.environ["AURA_SERVEUR"] == "1"

    def test_spec_serveur_ne_vas_pas_en_AURA_SPEC(self, tmp_path, monkeypatch):
        """Regression : AURA_SPEC (in-process) ralentit ; seul le serveur
        doit recevoir AURA_SERVEUR_SPEC."""
        monkeypatch.delenv("AURA_SPEC", raising=False)
        monkeypatch.delenv("AURA_SERVEUR_SPEC", raising=False)
        (tmp_path / "aura.toml").write_text(
            "[serveur]\nspec = false\n", encoding="utf-8")
        config.appliquer(chemin=tmp_path / "aura.toml")
        assert os.environ["AURA_SERVEUR_SPEC"] == "0"
        assert "AURA_SPEC" not in os.environ

    def test_un_budget_pinne_dans_le_fichier_ecrase_le_profil(self, tmp_path, monkeypatch):
        """Le cas reel du 8 oct. 2026, rendu visible.

        `aura.toml` epingle `budget_s = 45` ; le profil `qualite` veut 180 s
        pour rendre les dissertations completes. Le fichier gagne (c'est
        voulu) — mais le silence, lui, ne l'etait pas : c'est ce que
        `conflits_profil` rend visible.
        """
        monkeypatch.delenv("AURA_PROFIL", raising=False)
        chemin = tmp_path / "aura.toml"
        chemin.write_text("[profil]\nnom = \"qualite\"\n\n[qualite]\nbudget_s = 45\n",
                          encoding="utf-8")
        assert config.conflits_profil(chemin=chemin) == {"AURA_BUDGET_S": ("45", "180")}

    def test_l_argument_profil_gagne_sur_le_fichier(self, tmp_path, monkeypatch):
        """Priorite documentee : argument > fichier.

        Regression mesuree le 8 oct. 2026 : le fichier reecrivait AURA_PROFIL
        a l'etape « valeurs explicites », donc `--profil qualite` etait annule
        en silence par `[profil] nom = "equilibre"`.
        """
        monkeypatch.delenv("AURA_PROFIL", raising=False)
        monkeypatch.delenv("AURA_BUDGET_S", raising=False)
        chemin = tmp_path / "aura.toml"
        chemin.write_text("[profil]\nnom = \"equilibre\"\n", encoding="utf-8")
        pose = config.appliquer(chemin=chemin, profil="qualite")
        assert pose["AURA_PROFIL"] == "qualite"
        assert os.environ["AURA_PROFIL"] == "qualite"
        # Sans argument, le fichier choisit toujours : comportement conserve.
        monkeypatch.delenv("AURA_PROFIL", raising=False)
        assert config.appliquer(chemin=chemin)["AURA_PROFIL"] == "equilibre"

    def test_le_conflit_est_journalise(self, tmp_path, monkeypatch, caplog):
        """Un avertissement sans le remede ne sert a rien : on exige les deux."""
        import logging
        monkeypatch.delenv("AURA_PROFIL", raising=False)
        monkeypatch.delenv("AURA_BUDGET_S", raising=False)
        chemin = tmp_path / "aura.toml"
        chemin.write_text("[profil]\nnom = \"qualite\"\n\n[qualite]\nbudget_s = 45\n",
                          encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="aura.config"):
            config.appliquer(chemin=chemin)
        assert "ecrase le profil qualite" in caplog.text
        assert "retire" in caplog.text

    def test_aucun_conflit_quand_le_fichier_suit_le_profil(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AURA_PROFIL", raising=False)
        chemin = tmp_path / "aura.toml"
        chemin.write_text("[profil]\nnom = \"qualite\"\n\n[qualite]\nbudget_s = 180\n",
                          encoding="utf-8")
        assert config.conflits_profil(chemin=chemin) == {}
        # Un profil sans valeur pour la cle n'est pas un conflit non plus.
        chemin.write_text("[serveur]\nactif = true\n", encoding="utf-8")
        assert config.conflits_profil(chemin=chemin, profil="equilibre") == {}

    def test_version_lue_dans_pyproject(self):
        assert config.version().count(".") >= 1
        assert config.version() != "0.0.0"

    def test_profil_inconnu_retombe_sur_equilibre(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AURA_PROFIL", raising=False)
        pose = config.appliquer(chemin=tmp_path / "x.toml", profil="inconnu")
        assert pose["AURA_PROFIL"] == "equilibre"


# ── API locale /v1/* ───────────────────────────────────────────────────

class TestApi:
    @pytest.fixture()
    def serveur(self):
        from aura import api as api_mod
        etat = api_mod._Etat(smoke=True)          # jamais de modele en CI
        from http.server import ThreadingHTTPServer
        srv = ThreadingHTTPServer(("127.0.0.1", 0),
                                  api_mod._creer_handler(etat))
        fil = threading.Thread(target=srv.serve_forever, daemon=True)
        fil.start()
        yield f"http://127.0.0.1:{srv.server_address[1]}"
        srv.shutdown()

    @staticmethod
    def _get(base: str, route: str):
        try:
            with urllib.request.urlopen(base + route, timeout=10) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    @staticmethod
    def _post(base: str, route: str, corps: dict):
        req = urllib.request.Request(
            base + route, data=json.dumps(corps).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    def test_health(self, serveur):
        code, rep = self._get(serveur, "/v1/health")
        assert code == 200 and rep["status"] == "ok"

    def test_status_sans_modele(self, serveur):
        code, rep = self._get(serveur, "/v1/status")
        assert code == 200 and rep["prete"] is False

    def test_config_exposee(self, serveur):
        # l'import du module a pose la config UNE fois ; la fixture d'isolation
        # peut l'avoir retiree depuis -> on la repose ici (elle est restauree)
        config.appliquer()
        code, rep = self._get(serveur, "/v1/config")
        assert code == 200 and "AURA_PROFIL" in rep["environnement"]

    def test_question_vide_400(self, serveur):
        code, rep = self._post(serveur, "/v1/ask", {"question": ""})
        assert code == 400 and rep["ok"] is False

    def test_ask_sans_cerveau_503(self, serveur):
        code, rep = self._post(serveur, "/v1/ask", {"question": "bonjour"})
        assert code == 503 and rep["ok"] is False

    def test_route_inconnue_404(self, serveur):
        code, _ = self._get(serveur, "/v1/nimporte")
        assert code == 404


# ── auto-update ────────────────────────────────────────────────────────

class TestMaj:
    def test_normalisation_des_tags(self):
        assert maj._norme("v0.2.0") == (0, 2, 0)
        assert maj._norme("0.10") == (0, 10)
        assert maj._norme("") == (0,)
        assert maj._norme("beta") == (0,)

    def test_comparaison(self):
        assert maj._norme("v0.2.0") > maj._norme("0.1.0")
        assert maj._norme("v0.1.0") == maj._norme("0.1.0")
        assert not maj._norme("0.1.0") > maj._norme("v0.2.0")

    def test_desactivation_totale(self, monkeypatch):
        monkeypatch.setenv("AURA_MAJ", "0")
        rap = maj.verifier(timeout=0.1)
        assert rap["dispo"] is False and rap.get("desactive") is True

    def test_hors_ligne_ne_leve_jamais(self, monkeypatch):
        # point inatteignable -> echec silencieux (pas d'exception)
        monkeypatch.setattr(maj, "_URL", "http://127.0.0.1:1/releases")
        monkeypatch.delenv("AURA_MAJ", raising=False)
        rap = maj.verifier(timeout=0.2)
        assert rap["dispo"] is False
        assert "erreur" in rap or "nouvelle" in rap
