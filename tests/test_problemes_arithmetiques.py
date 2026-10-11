"""Problemes a deux etapes : fractions en mots et figures simples.

Deux familles trouvees par le SOUS-BANC (`scripts/sous_banc.py`) : elles
tombaient sur le LLM (~40 s, valeur non prouvee) alors qu'une seule
quantite suffit a les resoudre exactement — c'est justement pour ca que
les solveurs a une etape (filtre_instantane, calcul-verbal, calcul-taux)
ne les voyaient pas.

Le verrou qui compte ici n'est pas de repondre : c'est de **s'abstenir**
des que la forme n'est pas exactement celle attendue. Un solveur qui
devine est pire qu'un LLM, parce qu'il a l'air d'avoir raison.
"""
from aura import problemes_arithmetiques as P


class TestFractions:
    def test_deux_tiers(self):
        assert P.resoudre(
            "Un reservoir de 60 litres est rempli aux deux tiers. Combien de "
            "litres contient-il ?") == "40"

    def test_calcul_exact_sans_arrondi(self):
        """« deux tiers de 60 » doit valoir 40 PILE.

        En flottant, 60 * (2/3) donne 39,99999999999999 : la reponse
        exacte serait livree arrondie. D'ou `Fraction`.
        """
        assert P.resoudre("Les deux tiers de 60, combien ?") == "40"
        assert P.resoudre("Un tiers de 60, combien ?") == "20"
        assert P.resoudre("Les trois quarts de 80, combien ?") == "60"

    def test_fraction_sans_quantite_abstention(self):
        assert P.resoudre("Qu'est-ce que la moitie ?") is None

    def test_deux_nombres_abstention(self):
        """Deux nombres = deux etapes : terrain de calcul_verbal.

        « 12 bonbons, la moitie puis 2 » finit a 4 ; un solveur qui
        multiplierait par 1/2 rendrait 6 en ayant l'air d'avoir raison.
        """
        assert P.resoudre(
            "Alice a 12 bonbons, elle en donne la moitie a Bob puis 2 a "
            "Clara. Combien lui en reste-t-il ?") is None


class TestGeometrie:
    RECT = ("Un rectangle mesure 7 cm de long et 4 cm de large. ")

    def test_perimetre(self):
        assert P.resoudre(self.RECT + "Quel est son perimetre en cm ?") == "22"

    def test_aire(self):
        assert P.resoudre(
            self.RECT + "Quelle est son aire en cm carres ?") == "28"

    def test_nature_geometrique_absente_abstention(self):
        assert P.resoudre(self.RECT + "Quelle est sa surface ?") is None

    def test_unites_melangees_abstention(self):
        """7 cm et 4 m : la reponse dependrait de l'unite choisie."""
        assert P.resoudre(
            "Un rectangle de 7 cm de long et 4 m de large. Quel est son "
            "perimetre ?") is None

    def test_pas_de_figure_abstention(self):
        assert P.resoudre(
            "Une pizza de 8 parts est partagee entre 4 personnes. Combien de "
            "parts chacune ?") is None


class TestBranchement:
    def test_les_trois_questions_ne_generent_rien(self, monkeypatch):
        """Bout en bout : aucun de ces trois calculs ne doit atteindre le LLM."""
        from aura import llama_cerveau, filtre_instantane, rag_binaire
        from aura.orchestrateur import RontoGPT

        monkeypatch.setattr(
            llama_cerveau, "generer",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("problemes-arithmetiques doit repondre")))
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)

        ia = RontoGPT()
        for question, attendu in (
                (TestGeometrie.RECT + "Quel est son perimetre en cm ?", "22"),
                ("Un reservoir de 60 litres est rempli aux deux tiers. "
                 "Combien de litres contient-il ?", "40")):
            r = ia.executer_detaille(question)
            assert r["cerveau_choisi"] == "problemes-arithmetiques"
            assert r["reponse"] == attendu
