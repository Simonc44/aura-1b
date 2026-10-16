"""Tests du superviseur d'API (ex-`main.py` de la racine).

Ce code n'etait couvert par RIEN quand il vivait dans un `main.py` hors paquet
(aucun test, aucun doc ne le citait) : on verrouille ici le port transmis, la
reutilisation d'une API vivante, l'echec de demarrage, la relance avec
abandon, et le branchement des options de la CLI.
"""
import subprocess
import sys

import pytest

from aura import superviseur, tui as tui_module
from aura.__main__ import main as cli_main


class _FauxProcessus:
    """Popen minimal : un code de sortie par appel a `poll()`."""

    def __init__(self, codes):
        self._codes = list(codes)
        self.returncode = None
        self.termine = 0
        self.tue = 0

    def poll(self):
        if self._codes:
            self.returncode = self._codes.pop(0)
        return self.returncode

    def terminate(self):
        self.termine += 1
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.tue += 1


class _FauxSubprocess:
    """Remplace `superviseur.subprocess` : capture commande + environnement."""

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, a_rendre=None):
        self.appels: list[dict] = []
        self.a_rendre = a_rendre

    def Popen(self, args, env=None, **kw):
        self.appels.append({"args": list(args), "env": env})
        return self.a_rendre


@pytest.fixture
def faux_popen(monkeypatch):
    faux = _FauxSubprocess()
    monkeypatch.setattr(superviseur, "subprocess", faux)
    return faux


class TestSante:
    def test_repond_oui(self, monkeypatch):
        class _Rep:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        monkeypatch.setattr(superviseur.urllib.request, "urlopen",
                           lambda *a, **k: _Rep())
        assert superviseur.sante_ok() is True

    def test_repond_non_sur_erreur(self, monkeypatch):
        def boum(*_a, **_k):
            raise OSError("connexion refusee")

        monkeypatch.setattr(superviseur.urllib.request, "urlopen", boum)
        assert superviseur.sante_ok() is False


class TestDemarrage:
    def test_commande_et_port_transmis(self, faux_popen, monkeypatch):
        faux_popen.a_rendre = _FauxProcessus([None])
        monkeypatch.setenv("AURA_API_PORT", str(superviseur.PORT))
        superviseur.demarrer_api()
        appel = faux_popen.appels[0]
        assert appel["args"] == [sys.executable, "-m", "aura.api"]
        assert appel["env"]["AURA_API_PORT"] == str(superviseur.PORT)
        # l'environnement est une COPIE : rien n'est modifie chez l'appelant
        assert superviseur.os.environ.get("AURA_API_PORT") == str(superviseur.PORT)


class TestLancer:
    def test_reutilise_une_api_vivante(self, monkeypatch):
        monkeypatch.setattr(superviseur, "sante_ok", lambda: True)
        monkeypatch.setattr(superviseur, "demarrer_api",
                            lambda: pytest.fail("API deja vivante : ne pas en "
                                                "demarrer une seconde"))
        monkeypatch.setattr(tui_module, "lancer", lambda: 6)
        assert superviseur.lancer() == 6

    def test_api_muette_apres_demarrage_arrete_tout(self, monkeypatch):
        proc = _FauxProcessus([None])
        monkeypatch.setattr(superviseur, "sante_ok", lambda: False)
        monkeypatch.setattr(superviseur, "demarrer_api", lambda: proc)
        monkeypatch.setattr(superviseur, "attendre_sante", lambda *a: False)
        monkeypatch.setattr(tui_module, "lancer",
                            lambda: pytest.fail("rien ne s'ouvre si l'API est muette"))
        assert superviseur.lancer() == 1
        assert proc.termine == 1

    def test_interface_ouverte_puis_api_arretee(self, monkeypatch):
        proc = _FauxProcessus([None, None])
        monkeypatch.setattr(superviseur, "sante_ok", lambda: False)
        monkeypatch.setattr(superviseur, "demarrer_api", lambda: proc)
        monkeypatch.setattr(superviseur, "attendre_sante", lambda *a: True)
        monkeypatch.setattr(tui_module, "lancer", lambda: 3)
        assert superviseur.lancer() == 3
        assert proc.termine == 1           # l'API ne survit pas au TUI

    def test_api_seule_surveille(self, monkeypatch):
        proc = _FauxProcessus([None])
        vu = []
        monkeypatch.setattr(superviseur, "sante_ok", lambda: False)
        monkeypatch.setattr(superviseur, "demarrer_api", lambda: proc)
        monkeypatch.setattr(superviseur, "attendre_sante", lambda *a: True)
        monkeypatch.setattr(superviseur, "surveiller", lambda p: vu.append(p))
        monkeypatch.setattr(tui_module, "lancer",
                            lambda: pytest.fail("--api : aucune interface"))
        assert superviseur.lancer(interface=False) == 0
        assert vu == [proc]

    def test_api_seule_avec_api_deja_lancee(self, monkeypatch):
        monkeypatch.setattr(superviseur, "sante_ok", lambda: True)
        monkeypatch.setattr(superviseur, "demarrer_api",
                            lambda: pytest.fail("deja lancee"))
        monkeypatch.setattr(superviseur, "_attendre_interruption", lambda: None)
        monkeypatch.setattr(tui_module, "lancer",
                            lambda: pytest.fail("--api : aucune interface"))
        assert superviseur.lancer(interface=False) == 0


class TestArret:
    def test_processus_deja_mort_non_touche(self):
        proc = _FauxProcessus([0])
        superviseur.arreter(proc)
        assert proc.termine == 0

    def test_none_accepte(self):
        superviseur.arreter(None)          # ne leve pas


class TestSurveillance:
    @staticmethod
    def _faux_temps(monkeypatch, leve_au_bout_de: int | None = None):
        """`time` factice : les sleep ne dorment pas (et levent si demande)."""
        appels = []

        class _Temps:
            @staticmethod
            def time():
                return 0.0

            @staticmethod
            def sleep(_s):
                appels.append(_s)
                if leve_au_bout_de is not None and len(appels) >= leve_au_bout_de:
                    raise KeyboardInterrupt

        monkeypatch.setattr(superviseur, "time", _Temps)
        return appels

    def test_relance_apres_plantage(self, monkeypatch):
        # 1er poll : morte (code 1) -> une relance, puis Ctrl+C pendant l'attente
        self._faux_temps(monkeypatch, leve_au_bout_de=2)
        proc = _FauxProcessus([1])
        relances = []
        monkeypatch.setattr(superviseur, "demarrer_api",
                            lambda: relances.append(1) or _FauxProcessus([None]))
        monkeypatch.setattr(superviseur, "attendre_sante", lambda *a: True)
        with pytest.raises(KeyboardInterrupt):
            superviseur.surveiller(proc)
        assert len(relances) == 1

    def test_abandon_apres_le_plafond(self, monkeypatch):
        self._faux_temps(monkeypatch)          # sleep instantane
        relances = []

        def demarrer():
            relances.append(1)
            return _FauxProcessus([1])         # chaque relance meurt aussitot

        monkeypatch.setattr(superviseur, "demarrer_api", demarrer)
        monkeypatch.setattr(superviseur, "attendre_sante", lambda *a: False)
        superviseur.surveiller(_FauxProcessus([1]))     # rend la main : borne
        assert len(relances) == superviseur.MAX_RELANCES


class TestCli:
    def test_superviser_branche_le_superviseur(self, monkeypatch):
        vus = []
        monkeypatch.setattr(superviseur, "lancer",
                            lambda interface=True: vus.append(interface) or 0)
        monkeypatch.setattr(tui_module, "lancer",
                            lambda: pytest.fail("--superviser pilote lui-meme l'interface"))
        assert cli_main(["--superviser"]) == 0
        assert vus == [True]

    def test_superviser_api_seule(self, monkeypatch):
        vus = []
        monkeypatch.setattr(superviseur, "lancer",
                            lambda interface=True: vus.append(interface) or 0)
        assert cli_main(["--superviser", "--api"]) == 0
        assert vus == [False]
