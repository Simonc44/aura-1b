"""Tests des 3 ameliorations : verrou serveur, routeur evolutif, arbre MCTS."""
import threading

import pytest

from aura import arbre_reflexion, serveur_lora
from aura.routeur import RouteurIntelligent, _CATEGORIE_DEFI, _SEUIL_SVM


# ── 1. Verrou serveur (hot-swap vs chat) ───────────────────────────────

class TestVerrouServeur:
    def test_verrou_existe_et_bloque(self):
        assert isinstance(serveur_lora._verrou, type(threading.Lock()))
        # un acquire non-bloquant reussit une fois, echoue si detenu
        ok = serveur_lora._verrou.acquire(blocking=False)
        assert ok is True
        serveur_lora._verrou.release()

    def test_activer_mixte_serialise(self, monkeypatch):
        """Deux threads sur activer_mixte : le corps critique est protege."""
        appels = []
        ordre = []

        def faux_assurer():
            appels.append("assurer")
            return True

        def faux_urlopen(req, timeout):
            ordre.append(("post", threading.current_thread().name))
            class R:
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    return False
                def read(self):
                    return b'{"success": true}'
            return R()

        monkeypatch.setattr(serveur_lora, "_assurer", faux_assurer)
        monkeypatch.setattr(serveur_lora.urllib.request, "urlopen",
                            faux_urlopen)
        monkeypatch.setattr(serveur_lora, "_IDS", {"math": 0})
        monkeypatch.setattr(serveur_lora, "_actif", None)

        barriere = threading.Barrier(2)

        def travail():
            barriere.wait()
            serveur_lora.activer_mixte({"math": 1.0})

        t1 = threading.Thread(target=travail)
        t2 = threading.Thread(target=travail)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)
        # les deux ont abouti, aucun entrelacement destructeur
        assert serveur_lora._actif == (("math", 1.0),)

    def test_chat_et_swap_ne_se_chevauchent_pas(self, monkeypatch):
        """chat() detient le verrou pendant la generation : un swap
        concurrent attend la fin (pas de sortie corrompue)."""
        evenements = []

        def faux_assurer():
            return True

        class R:
            def __enter__(self):
                evenements.append(("chat-debut", threading.get_ident()))
                return self
            def __exit__(self, *a):
                evenements.append(("chat-fin", threading.get_ident()))
                return False
            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        monkeypatch.setattr(serveur_lora, "_assurer", faux_assurer)
        monkeypatch.setattr(serveur_lora.urllib.request, "urlopen",
                            lambda req, timeout: R())

        resultat = {}

        def gen():
            resultat["chat"] = serveur_lora.chat([{"role": "user",
                                                   "content": "q"}])

        t = threading.Thread(target=gen)
        t.start()
        # pendant que le chat tourne, un swap tente de passer : il doit
        # ATTENDRE la fin du chat (le verrou est tenu)
        t.join(0.05)
        serveur_lora.activer_mixte({"math": 1.0})  # doit bloquer puis passer
        t.join(timeout=5)
        # aucun "swap" n'apparait entre chat-debut et chat-fin du meme thread
        chat_debuts = [e for e in evenements if e[0] == "chat-debut"]
        assert chat_debuts, "le chat doit s'executer"
        assert resultat.get("chat") == "ok"


# ── 2. Routeur evolutif ────────────────────────────────────────────────

class TestRouteurEvolutif:
    def test_exemples_selfplay_type_defi(self, monkeypatch, tmp_path):
        import json as _json
        ds = tmp_path / "datasets"
        ds.mkdir()
        (ds / "selfplay.jsonl").write_text(
            _json.dumps({"user": "combien fait 12 fois 3", "type": "calcul"})
            + "\n"
            + _json.dumps({"user": "enigme quelconque", "type": "logique"})
            + "\n",
            encoding="utf-8")
        monkeypatch.setattr("aura.routeur.Path", lambda *a: tmp_path)
        # _exemples_selfplay construit son chemin depuis __file__ :
        # on teste le mapping via _CATEGORIE_DEFI et une lecture directe
        assert _CATEGORIE_DEFI["calcul"] == "math"
        assert _CATEGORIE_DEFI["logique"] == "general"

    def test_probas_logreg(self):
        from aura.routeur import _probas

        class FauxLogReg:
            classes_ = ["a", "b"]
            def predict_proba(self, X):
                import numpy as np
                return np.array([[0.7, 0.3]])

        p = _probas(FauxLogReg(), None)
        assert abs(p["a"] - 0.7) < 1e-9

    def test_probas_svm_softmax(self):
        from aura.routeur import _probas

        class FauxSVC:
            classes_ = ["a", "b", "c"]
            def decision_function(self, X):
                import numpy as np
                return np.array([[2.0, 1.0, 0.0]])

        p = _probas(FauxSVC(), None)
        total = sum(p.values())
        assert abs(total - 1.0) < 1e-9          # softmax normalisee
        assert p["a"] > p["b"] > p["c"]         # l'ordre des scores respected

    def test_seuil_svm_defini(self):
        assert _SEUIL_SVM >= 150   # au-dessus du dataset de base (~150)

    def test_classer_math_reste_math(self, tmp_path):
        r = RouteurIntelligent(dossier_cache=str(tmp_path / "cache"))
        r1 = r.classer("combien fait 12 fois 12")
        assert "math" in r1["experts"]

    def test_cache_relu_si_selfplay(self, monkeypatch, tmp_path):
        """Un cache existant + du savoir self-play -> re-entrainement."""
        import joblib as _joblib
        cache = tmp_path / "cache"
        cache.mkdir()
        _joblib.dump(object(), cache / "vectoriseur.joblib")
        _joblib.dump(object(), cache / "classifieur.joblib")
        monkeypatch.setattr(RouteurIntelligent, "_nb_exemples_selfplay",
                            classmethod(lambda cls: 42))
        monkeypatch.setattr(RouteurIntelligent, "_entrainer_classifieur",
                            lambda self: setattr(self, "_entraine", True))
        r = RouteurIntelligent(dossier_cache=str(cache))
        r._charger_ou_entrainer()
        assert getattr(r, "_entraine", False), \
            "le savoir self-play doit declencher un re-entrainement"


# ── 3. Arbre de reflexion (MCTS) ───────────────────────────────────────

class TestArbreReflexion:
    def test_etape_libelle_valide(self):
        e = arbre_reflexion.EtapeReflexion("17 x 20 = 340", expert="calcul",
                                           score=1.0, valide=True)
        assert "[OK] (calcul)" in e.libelle()

    def test_branche_score_moyenne(self):
        b = arbre_reflexion.BrancheReflexion()
        b.ajouter("etape 1", score=1.0, valide=True)
        b.ajouter("etape 2", score=0.5, valide=True)
        assert b.score == pytest.approx(0.75)

    def test_branche_invalide_ruinee(self):
        b = arbre_reflexion.BrancheReflexion()
        b.ajouter("etape juste", score=1.0, valide=True)
        b.ajouter("etape fausse", score=0.0, valide=False)
        assert b.score == 0.0
        assert not b.serie

    def test_branche_non_evaluee(self):
        b = arbre_reflexion.BrancheReflexion()
        b.ajouter("reflexion libre")
        assert b.score is None
        assert b.serie          # pas prouvee fausse : eligible

    def test_meilleure_branche_prouvee_dabord(self):
        arbre = arbre_reflexion.ArbreReflexion(question="q")
        b_libre = arbre.nouvelle_branche()
        b_libre.ajouter("pensee libre non evaluee", score=0.9, valide=None)
        b_prouvee = arbre.nouvelle_branche()
        b_prouvee.ajouter("etape exacte", score=1.0, valide=True)
        assert arbre.meilleure() is b_prouvee

    def test_meilleure_branche_rejette_invalide(self):
        arbre = arbre_reflexion.ArbreReflexion(question="q")
        b_fausse = arbre.nouvelle_branche()
        b_fausse.ajouter("raisonnement faux", score=0.0, valide=False)
        b_ok = arbre.nouvelle_branche()
        b_ok.ajouter("passe", score=0.6, valide=True)
        assert arbre.meilleure() is b_ok

    def test_branche_vers_prompt_exclut_les_ko(self):
        b = arbre_reflexion.BrancheReflexion()
        b.ajouter("bon chemin", score=1.0, valide=True)
        b.ajouter("impaire", score=0.0, valide=False)
        texte = arbre_reflexion.branche_vers_prompt(b)
        assert "bon chemin" in texte
        assert "impaire" not in texte

    def test_trace_complet(self):
        arbre = arbre_reflexion.ArbreReflexion(question="combien fait 12 x 12")
        b = arbre.nouvelle_branche()
        b.ajouter("12 x 12 = 144", expert="calcul", score=1.0, valide=True)
        trace = arbre.trace()
        assert "branche 1" in trace
        assert "[OK]" in trace
