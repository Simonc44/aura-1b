"""Tests de l'expert calcul d'achat (budget - prix x quantite).

Defaut reel mesure par scripts/banc_100.py : sur un probleme a DEUX
etapes, le 1B additionnait le billet au lieu de le soustraire
(« 12 euros + 20 euros = 32 euros » au lieu de 8). Le programme enchainE.
"""
from aura import calcul_achat
from aura.orchestrateur import RontoGPT


class TestCalculAchat:
    def test_reste_sur_budget(self):
        # etait 22 (le verbe « achete » etait compte comme une addition)
        assert calcul_achat.resoudre(
            "J'ai 15 euros et j'achete un livre a 7 euros, combien me "
            "reste-t-il ?") == "8"

    def test_prix_chacun(self):
        # etait 103 (100 + 3) : au banc, attendu 46
        assert calcul_achat.resoudre(
            "J'ai 100 euros. J'achete 3 livres a 18 euros chacun. Combien "
            "me reste-t-il ?") == "46"

    def test_monnaie_rendue(self):
        # etait « 12 euros + 20 euros = 32 euros » : au banc, attendu 8
        assert calcul_achat.resoudre(
            "Un cahier coute 3 euros. J'en achete 4 et je paie avec un "
            "billet de 20 euros. Combien me rend-on ?") == "8"

    def test_prix_au_kilo(self):
        assert calcul_achat.resoudre(
            "Un magasin vend des pommes a 2 euros le kilo. J'en achete 3 "
            "kilos et je paie avec 10 euros. Combien me rend-on ?") == "4"

    def test_quantite_et_prix_ensemble(self):
        assert calcul_achat.resoudre(
            "J'achete 4 stylos a 2 euros chacun, je paie avec 10 euros, "
            "combien me rend-on ?") == "2"

    def test_abstention_sans_monnaie(self):
        assert calcul_achat.resoudre("Combien font 17 x 23 ?") is None

    def test_abstention_sans_budget(self):
        # une depense sans budget annonce n'est pas un probleme d'achat
        assert calcul_achat.resoudre(
            "J'ai 10 euros et je depense 4 euros, combien me reste-t-il ?"
        ) is None

    def test_abstention_si_rend_plus_quon_a(self):
        assert calcul_achat.resoudre(
            "J'ai 5 euros, j'achete 3 cahiers a 4 euros, combien me "
            "reste-t-il ?") is None

    def test_total_de_plusieurs_articles(self):
        # etait 173 caracteres de bavardage au lieu de 16
        assert calcul_achat.resoudre(
            "J'achete 3 stylos a 2 euros et 2 cahiers a 5 euros. Combien "
            "je paie ?") == "16"

    def test_branche_dans_le_pipeline(self, monkeypatch):
        """L'achat ne doit JAMAIS atteindre le cerveau."""
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("calcul d'achat doit repondre")))
        r = RontoGPT().executer_detaille(
            "Un cahier coute 3 euros. J'en achete 4 et je paie avec un "
            "billet de 20 euros. Combien me rend-on ?")
        assert r["cerveau_choisi"] == "calcul-achat"
        assert r["reponse"] == "8"

    def test_ninterfere_pas_avec_calcul_verbal(self):
        """Un probleme raconte sans argent reste a calcul_verbal."""
        from aura import calcul_achat as ca, calcul_verbal as cv
        q = "J'ai 12 bonbons, j'en donne 5, combien m'en reste-t-il ?"
        assert ca.resoudre(q) is None
        assert cv.resoudre(q) == "7"
