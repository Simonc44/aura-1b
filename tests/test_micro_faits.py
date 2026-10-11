"""Micro-faits : le savoir elementaire chiffre resolu sans le modele.

Quatre questions trouvaient le 1B au sous-banc du 7 oct. 2026 (factuel-08,
factuel-11, factuel-12, format-couleurs). Elles n'ont pas d'etapes : ce n'est
pas un probleme de raisonnement, c'est un probleme de RAPPEL — et un modele de
1 milliard de parametres quantifie rate ces rappels.

Le verrou qui compte ici n'est pas de repondre : c'est de **s'abstenir** sur
tout ce qui n'est pas exactement une de ces formes. Ce tier ne doit jamais
court-circuiter une question qu'il ne sait pas trancher (le savoir etendu vit
dans `rag-binaire`, verifie) : une abstention rend la main, une reponse fausse
la retire.
"""
from aura import micro_faits as M


class TestCeQueLeBancAttend:
    """Les quatre echecs mesures, mot pour mot."""

    def test_annee_bissextile(self):
        assert "366" in M.resoudre(
            "Combien de jours compte une annee bissextile ?")

    def test_cotes_d_un_hexagone(self):
        rep = M.resoudre("Combien de cotes a un hexagone ?")
        assert rep is not None and "6" in rep

    def test_photosynthese(self):
        rep = M.resoudre(
            "De quoi les plantes ont-elles besoin pour la photosynthese ?")
        assert rep is not None
        assert "lumiere" in rep.lower() and "chlorophylle" in rep.lower()

    def test_couleurs_primaires(self):
        rep = M.resoudre(
            "Donne trois couleurs primaires separees par des virgules.")
        assert rep is not None
        assert sum(c in rep.lower() for c in ("rouge", "bleu", "jaune")) >= 2

    def test_insensible_aux_accents_et_a_la_casse(self):
        assert "366" in M.resoudre(
            "COMBIEN DE JOURS COMPTE UNE ANNÉE BISSEXTILE ?")


class TestLAbstention:

    def test_une_figure_sans_question_de_comptage_ne_declenche_rien(self):
        assert M.resoudre("Dessine un hexagone.") is None
        assert M.resoudre("Le triangle est une figure a trois cotes.") is None

    def test_une_question_de_comptage_sans_motif_connu_rend_la_main(self):
        # « combien » mais aucun comptage de la table : surtout ne pas deviner.
        assert M.resoudre("Combien d'habitants compte Paris ?") is None
        assert M.resoudre("Combien de films a realise Kubrick ?") is None

    def test_un_polygone_inconnu_ne_devine_pas(self):
        # Le cercle n'a pas de cote : l'inclure serait inventer.
        assert M.resoudre("Combien de cotes a un cercle ?") is None

    def test_point_commun_et_vide(self):
        assert M.resoudre("Combien de cotes a un pentagone ?") is not None
        assert M.resoudre("Quelle est la capitale de la France ?") is None
        assert M.resoudre("") is None
        assert M.resoudre("   ") is None


class TestDurees:
    def test_bissextile_avant_annee_ordinaire(self):
        """La regle la plus specifique doit gagner, sinon 366 deviendrait 365."""
        assert "366" in M.resoudre("Combien de jours compte une annee bissextile ?")
        assert "365" in M.resoudre("Combien de jours compte une annee ?")

    def test_les_autres_comptages(self):
        attendus = {
            "Combien de jours dans une semaine ?": "7",
            "Combien de mois compte une annee ?": "12",
            "Combien de minutes dans une heure ?": "60",
            "Combien de secondes dans une minute ?": "60",
            "Combien d'heures compte une journee ?": "24",
        }
        for question, nombre in attendus.items():
            rep = M.resoudre(question)
            assert rep is not None and nombre in rep, (question, rep)

    def test_les_polygones(self):
        for nom, cotes in M.COTES_POLYGONES.items():
            rep = M.resoudre(f"Combien de cotes a un {nom} ?")
            assert rep is not None and str(cotes) in rep, nom
