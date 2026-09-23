"""Tests des experts deterministes calcul verbal et tri transitif.

Ces experts interceptent les petits problemes racontes que ni le
niveau 0 (pas d'expression mathematique) ni le mini-SAT (formalisme
trop lourd) ne savaient resoudre — le 1B repondait alors hors-sujet.
"""
from aura import calcul_verbal, tri_transitif
from aura.orchestrateur import Aura1B


class TestCalculVerbal:
    def test_pommes(self):
        assert calcul_verbal.resoudre(
            "si j ai 3 pommes et que j en mange 1 combien il m en reste") == "2"

    def test_argent_perdu(self):
        assert calcul_verbal.resoudre(
            "j avais 10 euros j en perds 4 combien me reste t il") == "6"

    def test_achats_enumeration(self):
        assert calcul_verbal.resoudre(
            "j achete 5 cahiers et 3 stylos combien en tout") == "8"

    def test_gain(self):
        assert calcul_verbal.resoudre(
            "j avais 12 billes j en gagne 7 combien en tout") == "19"

    def test_pas_de_question_pas_de_calcul(self):
        assert calcul_verbal.resoudre("qu'est ce que la photosynthese") is None
        assert calcul_verbal.resoudre("raconte moi une blague") is None

    def test_abstention_si_negatif(self):
        # 3 - 5 = -2 : quantite physique absurde -> abstention (None),
        # le cerveau repondra — le traducteur n'invente pas.
        assert calcul_verbal.resoudre(
            "j ai 3 pommes j en mange 5 combien il m en reste") is None

    def test_branche_dans_le_pipeline(self, monkeypatch):
        """La question pommes ne doit JAMAIS atteindre le cerveau."""
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("calcul verbal doit repondre")))
        ia = Aura1B()
        r = ia.executer_detaille(
            "si j ai 3 pommes et que j en mange 1 combien il m en reste")
        assert r["cerveau_choisi"] == "calcul-verbal"
        assert r["reponse"] == "2"


class TestTriTransitif:
    def test_chaine_age(self):
        assert tri_transitif.repondre(
            "resous cette enigme logique : Paul est plus age que Marie, "
            "Marie est plus age que Leo, qui est le plus age ?") == "Paul"

    def test_signes(self):
        assert tri_transitif.repondre(
            "A > B, B > C, qui est le plus grand ?") == "A"

    def test_inversion_petit(self):
        assert tri_transitif.repondre(
            "Leo est plus jeune que Marie et Marie est plus jeune que Paul. "
            "Qui est le plus jeune ?") == "Leo"

    def test_une_seule_relation_insuffisante(self):
        assert tri_transitif.repondre(
            "Paul est plus age que Marie") is None

    def test_sans_question_pas_dexperte(self):
        assert tri_transitif.repondre(
            "qu est ce que la photosynthese") is None

    def test_branche_dans_le_pipeline(self, monkeypatch):
        """L'enigme doit etre resolue par le tri, pas par le 1B.

        Le cache niveau 0 est neutralise : sinon une execution precedente
        (test unitaire, question reelle) repond avant le pipeline —
        comportement voulu en prod, mais ce test verifie le ROUTAGE.
        """
        from aura import llama_cerveau, filtre_instantane
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("tri transitif doit repondre")))
        ia = Aura1B()
        r = ia.executer_detaille(
            "Paul est plus age que Marie, Marie est plus age que Leo, "
            "qui est le plus age ?")
        assert r["cerveau_choisi"] == "tri-transitif"
        assert r["reponse"] == "Paul"

    def test_cycle_abstention(self):
        # A > B et B > A : contradiction -> None (jamais de reponse au
        # hasard face a une enigme incoherente)
        assert tri_transitif.trier([("A", "B"), ("B", "A")]) is None
