"""Tests des 3 idees importees de JEV ultrafast (browser-use) :
1. cycle unique : niveau 0 + routeur en parallele (max des durees)
2. validation avant execution : garde-fou anti-myroutage
3. fan-out parallele : web ∥ PGS, echecs isoles
"""
import threading
import time

from aura import filtre_instantane, memoire_web
from aura.orchestrateur import Aura1B


def _ia_neutre(monkeypatch, analyse=None):
    """Aura sans GGUF requis, cache niveau 0 neutre, routage force en option."""
    from aura import llama_cerveau
    monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
    monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
    monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "ok-llm")
    monkeypatch.setattr(llama_cerveau, "generer_riche", lambda *a, **k: "ok-llm")
    if analyse is not None:
        monkeypatch.setattr(Aura1B, "analyser", lambda self, q: analyse)
    return Aura1B()


class TestCycleUnique:
    def test_hit_niveau0_nattend_pas_le_routeur(self, monkeypatch):
        """Un routeur lent (1 s) ne doit pas retarder un hit niveau 0."""
        def routeur_lent(self, q):
            time.sleep(1.0)
            return {"experts": {"general"}, "methodes": ["test"]}

        monkeypatch.setattr(Aura1B, "analyser", routeur_lent)
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: "81")
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        ia = Aura1B()
        t0 = time.perf_counter()
        r = ia.executer_detaille("Calcule le carre de 9")
        duree = time.perf_counter() - t0
        assert r["cerveau_choisi"] == "niveau0-instantane"
        assert r["reponse"] == "81"
        assert duree < 0.6        # un appel sequentiel ferait >= 1.0 s

    def test_miss_le_routeur_est_consulte(self, monkeypatch):
        ia = _ia_neutre(monkeypatch, analyse={"experts": {"general"},
                                              "methodes": ["test"]})
        r = ia.executer_detaille("bonjour")
        assert r["cerveau_choisi"] == "llama-3.2-1b"
        assert "general" in r["experts"]


class TestGardeFou:
    def test_math_sans_donnees_ignorer(self, monkeypatch):
        ia = _ia_neutre(monkeypatch, analyse={"experts": {"math"},
                                              "methodes": ["test"]})
        r = ia.executer_detaille("quel est le carre de 12")
        assert "math" not in r["experts"]
        assert "general" in r["experts"]
        assert r["formule"] == ""

    def test_math_seul_replie_sur_general(self, monkeypatch):
        ia = _ia_neutre(monkeypatch, analyse={"experts": {"math"},
                                              "methodes": ["test"]})
        r = ia.executer_detaille("resous cette integrale")
        assert r["experts"] == {"general"}

    def test_math_avec_donnees_conserve(self, monkeypatch):
        ia = _ia_neutre(monkeypatch, analyse={"experts": {"math"},
                                              "methodes": ["test"]})
        monkeypatch.setattr(ia, "resoudre_numerique", lambda X, y: "mul(X0,X0)")
        r = ia.executer_detaille("decouvre la loi", X=[[1.0], [2.0]],
                                 y=[1.0, 4.0])
        assert r["formule"] == "mul(X0,X0)"

    def test_donnees_valides(self):
        assert not Aura1B._donnees_valides(None, None)
        assert not Aura1B._donnees_valides([1.0], None)
        assert not Aura1B._donnees_valides([1.0], [1.0, 2.0])   # longueurs
        assert not Aura1B._donnees_valides([1.0], [1.0])        # 1 point seul
        assert Aura1B._donnees_valides([1.0, 2.0], [1.0, 4.0])


class TestFanOutParallele:
    def test_deux_experts_sur_threads_distincts(self, monkeypatch):
        ident = {}

        def web(q, **k):
            time.sleep(0.05)   # les 2 threads vivent en meme temps :
            ident["web"] = threading.get_ident()   # sans ca, un ident peut
            return "FAIT"                          # etre recycle (test flaky)

        def pgs(X, y):
            time.sleep(0.05)
            ident["pgs"] = threading.get_ident()
            return "mul(X0,X0)"

        ia = _ia_neutre(monkeypatch, analyse={"experts": {"web", "math"},
                                              "methodes": ["test"]})
        monkeypatch.setattr(memoire_web, "chercher", web)
        monkeypatch.setattr(ia, "resoudre_numerique", pgs)
        # question non-riche : web passe par chercher() directement
        r = ia.executer_detaille("prix de l or aujourd hui", X=[[1.0], [2.0]],
                                 y=[1.0, 4.0])
        assert r["contexte_web"] == "FAIT"
        assert r["formule"] == "mul(X0,X0)"
        principal = threading.main_thread().ident
        assert ident["web"] != principal and ident["pgs"] != principal
        assert ident["web"] != ident["pgs"]       # vraie parallelisation

    def test_web_et_pgs_se_recouvrent(self, monkeypatch):
        """2 taches de 0.3 s en parallele : < 0.58 s (sequentiel >= 0.6 s)."""
        ia = _ia_neutre(monkeypatch, analyse={"experts": {"web", "math"},
                                              "methodes": ["test"]})
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda q, **k: (time.sleep(0.3), "FAIT")[1])
        monkeypatch.setattr(ia, "resoudre_numerique",
                            lambda X, y: (time.sleep(0.3), "mul(X0,X0)")[1])
        t0 = time.perf_counter()
        # question non-riche : chercher() appele une seule fois
        r = ia.executer_detaille("prix de l or aujourd hui", X=[[1.0], [2.0]],
                                 y=[1.0, 4.0])
        duree = time.perf_counter() - t0
        assert r["contexte_web"] == "FAIT" and r["formule"] == "mul(X0,X0)"
        assert duree < 0.58

    def test_echec_web_nempeche_pas_pgs(self, monkeypatch):
        ia = _ia_neutre(monkeypatch, analyse={"experts": {"web", "math"},
                                              "methodes": ["test"]})

        def _boom(*a, **k):
            raise RuntimeError("reseau off")

        monkeypatch.setattr(memoire_web, "chercher", _boom)
        monkeypatch.setattr(ia, "resoudre_numerique", lambda X, y: "mul(X0,X0)")
        r = ia.executer_detaille("analyse", X=[[1.0], [2.0]], y=[1.0, 4.0])
        assert r["contexte_web"] == ""
        assert r["formule"] == "mul(X0,X0)"

    def test_un_seul_expert_appel_direct(self, monkeypatch):
        ia = _ia_neutre(monkeypatch, analyse={"experts": {"web"},
                                              "methodes": ["test"]})
        monkeypatch.setattr(memoire_web, "chercher", lambda q, **k: "FAIT")
        r = ia.executer_detaille("actualite")
        assert r["contexte_web"] == "FAIT"

    def test_riche_passe_par_chercher_enrichi(self, monkeypatch):
        ia = _ia_neutre(monkeypatch, analyse={"experts": {"web"},
                                              "methodes": ["test"]})
        vu = []
        monkeypatch.setattr(memoire_web, "chercher_enrichi",
                            lambda q, **k: vu.append(1) or "ENRICHI")
        r = ia.executer_detaille(
            "Redige une analyse comparee du solaire et de l'eolien en France")
        assert r["contexte_web"] == "ENRICHI" and vu
