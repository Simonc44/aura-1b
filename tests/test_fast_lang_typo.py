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


class TestRenduConjugaison:
    """Le RENDU (`repondre`), pas seulement la table.

    Angles morts reels : les tests ne couvraient que `get_conjugation` et
    `conjuguer_table` — les tables. Le rendu, lui, ecrivait
    `f"je {table['je']}"` sans elision, donc « conjugue avoir au futur »
    repondait « je aurai ». La table etait juste ; la phrase etait fausse ;
    et rien ne le voyait, parce que le banc ne pose aucune question de
    conjugaison (0/100).
    """

    def test_elision_devant_voyelle(self, moteur):
        r = moteur.repondre("conjugue avoir au futur")
        assert "j'aurai" in r, r
        assert "je aurai" not in r, r

    @pytest.mark.parametrize("question,attendu", [
        ("conjugue aimer au present", "j'aime"),
        ("conjugue ecouter au present", "j'écoute"),
    ])
    def test_elision_verbes_en_voyelle(self, moteur, question, attendu):
        assert attendu in moteur.repondre(question)

    @pytest.mark.parametrize("question,attendu", [
        ("conjugue etre au futur", "je serai"),
        ("conjugue aller au present", "je vais"),
        ("conjugue parler au present", "je parle"),
    ])
    def test_pas_d_elision_devant_consonne(self, moteur, question, attendu):
        r = moteur.repondre(question)
        assert attendu in r, r
        assert "j'" not in r, r

    def test_les_six_personnes_sont_rendues(self, moteur):
        r = moteur.repondre("conjugue parler au present")
        for forme in ("je parle", "tu parles", "il/elle parle",
                      "nous parlons", "vous parlez", "ils/elles parlent"):
            assert forme in r, (forme, r)

    def test_table_absente_ne_plante_pas(self, moteur):
        assert moteur.repondre("conjugue zxqv au present") is None


class TestCapaciteNonTesteeParLeBanc:
    """fast-lang est DECLARE au registre : sa couverture doit etre reelle.

    Le banc 100 mesure 0 question servie par ce tier — non parce qu'il est
    mort (il repond en < 0,2 ms), mais parce que le banc ne pose aucune
    question de langue. Une capacite que rien ne teste est une capacite qui
    casse en silence : c'est exactement ce qui etait arrive a l'elision.
    """

    def test_declare_au_registre_et_branche(self):
        from aura import registre_experts as R
        assert "fast-lang" in R.noms()
        assert R.expert("fast-lang").genre == R.EXACT

    def test_le_banc_naucune_question_de_conjugaison(self):
        import sys
        from pathlib import Path
        racine = Path(__file__).resolve().parent.parent
        for dossier in (racine, racine / "scripts"):
            if str(dossier) not in sys.path:
                sys.path.insert(0, str(dossier))
        import banc_100 as B
        assert not [q for q in B.QUESTIONS if "conjugu" in q[2].lower()], (
            "si le banc se met a couvrir la conjugaison, ce test doit etre "
            "remplace par la mesure de la couverture de fast-lang")


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
