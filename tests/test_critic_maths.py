"""Tests CRITIC (vérification outillée) + maths composées au niveau 0.

Les deux corrections des bugs trouvés en test à froid :
- « 15 divise par 3 plus 4 puissance 2 » → 16 (au lieu de 21)
- « capitale de l australie » → Sydney (au lieu de Canberra)
"""
import pytest

from aura import filtre_instantane
from aura.orchestrateur import Aura1B


class TestMathsComposees:
    """Le bug réel : les motifs simples attrapaient « 4 puissance 2 »
    avant de voir l'expression complète."""

    def test_le_bug_original(self):
        r = filtre_instantane._calcul_direct(
            "combien fait 15 divise par 3 plus 4 puissance 2")
        assert r == "21"

    def test_expression_simples(self):
        f = filtre_instantane._calcul_direct
        assert f("15 divise par 3") == "5"
        assert f("3 fois 7 plus 2") == "23"
        assert f("5 au carre moins 3") == "22"
        assert f("2 puissance 10") == "1 024"
        assert f("10 divise par 4") == "2.5"

    def test_parentheses_dictees(self):
        f = filtre_instantane._calcul_direct
        assert f("2 fois 3 plus 4") == "10"      # priorité ×  avant +

    def test_demande_math_sans_operateur_fort(self):
        f = filtre_instantane._calcul_direct
        # « combien fait 45 + 12 » : déjà géré, pas de régression
        assert f("calcule 45*12+3") == "543"

    def test_pas_de_fausse_conversion(self):
        """Le non-math ne doit PAS être converti ni calculé."""
        f = filtre_instantane._calcul_direct
        assert f("il fait moins 5 degres") is None
        assert f("raconte moi une blague") is None
        assert f("plus belle la vie") is None

    def test_un_seul_nombre_rejete(self):
        assert filtre_instantane._calcul_direct(
            "combien fait 42") is None

    def test_garde_fou_puissance(self):
        assert filtre_instantane._calcul_direct(
            "2 puissance 10000") is None


class TestVerificationCritic:
    def test_factuel_sans_web_declenche(self):
        assert Aura1B._a_besoin_verification(
            "capitale de l australie", "", "Canberra.")

    def test_deja_gelee_pas_de_verif(self):
        assert not Aura1B._a_besoin_verification(
            "capitale de l australie", "Titre : x | Extrait : y", "Canberra.")

    def test_riche_pas_de_verif(self):
        assert not Aura1B._a_besoin_verification(
            "explique pourquoi le ciel est bleu", "", "x" * 500)

    def test_contextuelle_pas_de_verif(self):
        assert not Aura1B._a_besoin_verification(
            "comment je m appelle", "", "Tu es Simon.")

    def test_non_factuel_pas_de_verif(self):
        assert not Aura1B._a_besoin_verification(
            "raconte moi une histoire", "", "Il était une fois...")

    def test_reponse_conservee_si_web_vide(self, monkeypatch):
        from aura import memoire_web
        monkeypatch.setattr(memoire_web, "chercher", lambda *a, **k: "")
        assert Aura1B._verifier_au_web("q", "rep") == "rep"

    def test_reponse_conservee_si_web_echoue(self, monkeypatch):
        from aura import memoire_web

        def _boom(*a, **k):
            raise RuntimeError("off")

        monkeypatch.setattr(memoire_web, "chercher", _boom)
        assert Aura1B._verifier_au_web("q", "rep") == "rep"

    def test_revision_si_contredite(self, monkeypatch):
        """Le web contredit le 1B -> la réponse est ré-ancrée."""
        from aura import llama_cerveau, memoire_web
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda *a, **k: "Canberra est la capitale")
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "Canberra.")
        assert Aura1B._verifier_au_web("capitale de l australie",
                                       "Sydney.") == "Canberra."

    def test_desactivable(self, monkeypatch):
        monkeypatch.setenv("AURA_VERIF_WEB", "0")
        assert not Aura1B._a_besoin_verification(
            "capitale de l australie", "", "Canberra.")

    def test_reconsolidation_cache(self, monkeypatch, tmp_path):
        """La réponse vérifiée REMPLACE l'ancienne au cache (pas de doublon).

        C'est le bug réel : « Sydney » mis en cache restait collé à vie
        car enregistrer() refuse les entrées similaires.
        """
        monkeypatch.setattr(filtre_instantane, "_FICHIER",
                            tmp_path / "cache.jsonl")
        monkeypatch.setattr(filtre_instantane, "_vectoriseur", None)
        monkeypatch.setattr(filtre_instantane, "_matrice", None)
        monkeypatch.setattr(filtre_instantane, "_entrees", [])
        filtre_instantane.enregistrer("capitale de l australie",
                                      "L'Australie a pour capitale Sydney.")
        # l'écriture est refusée en doublon : le cache garde le faux...
        filtre_instantane.enregistrer("capitale de l australie", "Canberra.")
        assert filtre_instantane.repondre("capitale de l australie") != "Canberra."
        # ...sauf via la reconsolidation
        assert filtre_instantane.mettre_a_jour("capitale de l australie",
                                               "Canberra.")
        assert filtre_instantane.repondre("capitale de l australie") == "Canberra."

    def test_integration_pipeline_critic(self, monkeypatch):
        """Pipeline complet : factuel LLM faux -> révisé par la preuve web."""
        from aura import filtre_instantane, llama_cerveau, memoire_web
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda *a, **k: "Canberra est la capitale de "
                                            "l'Australie depuis 1913.")
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda q, contexte_web="", formule="",
                            max_tokens=None, historique=None, systeme=None,
                            contexte_faits="", categorie="":
                            ("Canberra." if contexte_web else "Sydney."))
        ia = Aura1B()
        monkeypatch.setattr(
            ia, "analyser",
            lambda q: {"experts": {"general"}, "prediction": "general",
                       "scores": {"general": 0.70, "web": 0.15, "math": 0.15}})
        r = ia.executer_detaille("capitale de l australie")
        assert r["reponse"] == "Canberra."
