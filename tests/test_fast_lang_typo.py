"""Tests fast_lang (dictionnaire symbolique SQLite) + typographie
(post-processeur regex). La langue comme TABLES, pas comme génération.
"""
import pytest

from aura import fast_lang, typographie


@pytest.fixture(scope="module")
def moteur():
    if not fast_lang._BDD.exists():
        pytest.skip("lang.db absente (lancer scripts/construire_lang.py)")
    return fast_lang.FastLangEngine()


class TestConjugaison:
    def test_irregulier_present(self, moteur):
        assert moteur.get_conjugation("être", "present", "je") == "suis"
        assert moteur.get_conjugation("être", "futur", "je") == "serai"
        assert moteur.get_conjugation("avoir", "present", "ils") == "ont"

    def test_sans_accents(self, moteur):
        # l'utilisateur écrit « etre » sans accent : même table
        assert moteur.get_conjugation("etre", "futur", "nous") == "serons"

    def test_er_regulier(self, moteur):
        assert moteur.get_conjugation("parler", "present", "nous") == "parlons"
        assert moteur.get_conjugation("manger", "futur", "ils") == "mangeront"

    def test_table_complete(self, moteur):
        t = moteur.conjuguer_table("être", "imparfait")
        assert t and len(t) == 6 and t["je"] == "étais"

    def test_inconnu_renvoie_none(self, moteur):
        assert moteur.get_conjugation("zxqv", "present", "je") is None
        assert moteur.get_conjugation("être", "plusqueparfait", "je") is None

    def test_temps_invalide(self, moteur):
        assert moteur.get_conjugation("être", "subjonctif;;", "je") is None


class TestDefinition:
    def test_exacte(self, moteur):
        d = moteur.get_definition("photosynthèse")
        assert d and "lumière" in d

    def test_tolerant_aux_accents(self, moteur):
        assert moteur.get_definition("photosynthese") == \
            moteur.get_definition("photosynthèse")

    def test_inconnu_none(self, moteur):
        assert moteur.get_definition("xylophonequantique") is None


class TestElision:
    def test_voyelle(self, moteur):
        assert moteur.elision("le", "arbre") == "l'"
        assert moteur.elision("la", "école") == "l'"

    def test_h_aspire_garde(self, moteur):
        assert moteur.elision("le", "héros") == "le"
        assert moteur.elision("le", "hameau") == "le"

    def test_h_muet_elide(self, moteur):
        assert moteur.elision("le", "hérisson") == "l'"

    def test_consonne_non_concerne(self, moteur):
        assert moteur.elision("le", "chat") is None


class TestDetecteur:
    def test_conjugue(self, moteur):
        r = moteur.repondre("conjugue être au futur")
        assert r and "serai" in r and "seront" in r

    def test_conjugue_sans_accents(self, moteur):
        r = moteur.repondre("conjugue etre au present")
        assert r and "suis" in r

    def test_definition_connue(self, moteur):
        r = moteur.repondre("qu'est-ce que la photosynthèse")
        assert r and "lumière" in r

    def test_definition_inconnue_none(self, moteur):
        # pas d'invention : cascade normale
        assert moteur.repondre("qu'est-ce que la mécanique quantique") is None

    def test_question_ordinaire_none(self, moteur):
        assert moteur.repondre("combien font 7 fois 8") is None


class TestTypographie:
    def test_elision(self):
        # l'élision produit l'APOSTROPHE TYPOGRAPHIQUE (U+2019)
        assert typographie.nettoyer("le arbre") == "l’arbre"
        assert typographie.nettoyer("la école") == "l’école"

    def test_h_aspire_preserve(self):
        assert typographie.nettoyer("le héros") == "le héros"
        assert typographie.nettoyer("le hall") == "le hall"

    def test_nbsp(self):
        r = typographie.nettoyer("vraiment ?")
        assert "\u00A0?" in r
        r = typographie.nettoyer("liste : un, deux")
        assert "\u00A0:" in r

    def test_apostrophe_typographique(self):
        assert typographie.nettoyer("l'arbre") == "l’arbre"

    def test_idempotent(self):
        t = "le arbre, la école ? oui!"
        assert typographie.nettoyer(typographie.nettoyer(t)) == \
            typographie.nettoyer(t)

    def test_code_preserve(self):
        code = "print('le arbre ?')"
        brut = f"Voici ```python\n{code}\n``` le arbre"
        r = typographie.hors_code(brut)
        assert "print('le arbre ?')" in r       # code intact
        assert "l’arbre" in r                   # prose corrigée

    def test_vide(self):
        assert typographie.nettoyer("") == ""
