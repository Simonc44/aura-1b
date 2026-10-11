"""Solveur de chaines numeriques : l'algebre elementaire par le CALCUL.

Ces 4 questions de la categorie logique du banc partaient au 1B, qui
produisait une chaine fluente mais fausse (mode d'echec documente des petits
modeles). Ce sont des equations lineaires a une inconnue : ici elles sont
RESOLUES, pas racontees.

Le test le plus important du fichier est `test_relation_nancre_pas` : sans le
lookahead sur « de plus/moins », « Alice a 3 ans de plus que Bob » ancrerait
Alice = 3 et la propagation partirait d'une valeur fausse. Un solveur exact
qui se trompe est plus dangereux que pas de solveur du tout.
"""
import pytest

from aura import chaines_numeriques as C


class TestChainesDuBanc:
    @pytest.mark.parametrize("question,attendu", [
        ("Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que Clara. "
         "Clara a 4 ans. Quel age a Alice ?", "9"),
        ("Lea a 2 ans de moins que Tom. Tom a 5 ans de plus que Zoe. "
         "Zoe a 8 ans. Quel age a Lea ?", "11"),
        ("Marc a 4 ans de plus que Julie. Julie a 3 ans de moins que Sam. "
         "Sam a 10 ans. Quel age a Marc ?", "11"),
        ("Trois amis ont des ages differents : Ana 25 ans, Ben 31 ans, "
         "Chloe 28 ans. Qui est le plus jeune ?", "Ana"),
    ])
    def test_resolu_exactement(self, question, attendu):
        assert C.resoudre(question) == attendu

    def test_extremum_maximum(self):
        q = ("Ana 25 ans, Ben 31 ans, Chloe 28 ans. Qui est le plus age ?")
        assert C.resoudre(q) == "Ben"


class TestAbstentions:
    def test_relation_ne_cree_pas_d_ancre(self):
        """« Alice a 3 ans de plus que Bob » ne dit PAS qu'Alice a 3 ans."""
        r = C.resoudre(
            "Alice a 3 ans de plus que Bob. Quel age a Alice ?")
        assert r is None, f"ancre fausse creee par la relation : {r}"

    def test_valeur_manquante(self):
        """Chaine non ancree : impossible de connaitre la valeur absolue."""
        assert C.resoudre(
            "Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que Clara. "
            "Quel age a Alice ?") is None

    def test_unites_melangees(self):
        """Masse et duree ne se propagent pas ensemble."""
        assert C.resoudre(
            "Alice a 3 kg de plus que Bob. Bob a 2 ans de plus que Clara. "
            "Clara a 4 ans. Quel age a Alice ?") is None

    def test_pas_une_enigme(self):
        assert C.resoudre("Quelle heure est-il ?") is None
        assert C.resoudre("") is None

    def test_entite_ambigue(self):
        """Deux entites citees dans la question : on ne tranche pas."""
        assert C.resoudre(
            "Alice a 3 ans de plus que Bob. Clara a 4 ans. "
            "Quel age a Alice et Claire ?") is None

    def test_extremum_sans_valeurs(self):
        assert C.resoudre("Qui est le plus grand ?") is None


class TestDansLaCascade:
    def test_tier_declare_et_branche(self, monkeypatch):
        """Le tier est declare au registre ET branche dans la cascade."""
        from aura import registre_experts as R
        from aura.orchestrateur import RontoGPT
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "[GENERE]")
        assert "chaines-numeriques" in R.noms()
        r = RontoGPT().executer_detaille(
            "Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que Clara. "
            "Clara a 4 ans. Quel age a Alice ?")
        assert r["cerveau_choisi"] == "chaines-numeriques"
        assert r["reponse"] == "9"
