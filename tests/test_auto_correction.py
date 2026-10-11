"""Self-Correction Stream : ce qui est verifiable EXACTEMENT, et rien d'autre.

Le controle qui compte ici n'est pas « detecte-t-il des erreurs ? » mais
« se TAIT-il quand il n'y a rien de solide a dire ? ». Un controleur qui
accuse a tort est pire que pas de controleur : il coupe une generation
correcte et fait repondre le modele a cote.

D'ou les deux familles de tests :
  * des egalites FAUSSES (il doit parler) ;
  * des egalites justes, ambigues ou non calculables (il doit se taire).
"""
import pytest

from aura import auto_correction as ac


class TestEgalitesFausses:
    def test_addition_fausse(self):
        note = ac.verifier("2 + 2 = 5")
        assert note is not None and "4" in note

    def test_multiplication_fausse_symbole_x(self):
        assert ac.verifier("17 x 3 = 50") is not None

    def test_multiplication_fausse_symbole_unicode(self):
        assert ac.verifier("17 × 3 = 50") is not None

    def test_puissance_fausse(self):
        assert ac.verifier("2 ** 10 = 1023") is not None

    def test_soustraction_fausse(self):
        assert ac.verifier("9 - 4 = 6") is not None

    def test_la_note_donne_la_valeur_juste(self):
        note = ac.verifier("12 / 4 = 4")
        assert note is not None and "3" in note

    def test_egalite_noyee_dans_une_phrase(self):
        note = ac.verifier("Je calcule d'abord : 7 + 5 = 13. Ensuite je "
                           "poursuis le raisonnement.")
        assert note is not None and "12" in note


class TestIlDoitSeTaire:
    def test_egalite_juste(self):
        assert ac.verifier("17 x 3 = 51") is None

    def test_decimaux_exacts_pas_de_faux_positif(self):
        """0,1 + 0,2 = 0,3 : avec des flottants ce test ECHOUE.

        C'est la raison du choix des rationnels exacts : juger en flottant
        ferait accuser un modele qui a parfaitement raison.
        """
        assert ac.verifier("0.1 + 0.2 = 0.3") is None

    def test_separateur_de_milliers(self):
        assert ac.verifier("1 024 + 0 = 1 024") is None

    def test_division_par_zero_ne_plante_pas(self):
        assert ac.verifier("5 / 0 = 0") is None

    def test_puissance_absurde_bornee(self):
        """2^100000 n'est pas un controle, c'est un deni de service."""
        assert ac.verifier("2 ^ 100000 = 3") is None

    def test_texte_sans_egalite(self):
        assert ac.verifier("Je reflechis a la question posee.") is None

    def test_texte_vide(self):
        assert ac.verifier("") is None
        assert ac.verifier(None) is None

    def test_inegalite_non_jugee(self):
        """« 2 + 2 < 5 » n'est pas une egalite : hors du contrat."""
        assert ac.verifier("2 + 2 < 5") is None


class TestSurveillance:
    def test_alerte_detectee_en_flux(self):
        s = ac.SurveillancePensee()
        s.pousser("Je pose le calcul. ")
        s.pousser("Or 6 x 7 = 41. ")
        assert s.alerte is not None and "42" in s.alerte

    def test_virgule_ne_termine_pas_la_phrase(self):
        """Un membre droit peut suivre une virgule : ne pas juger trop tot."""
        s = ac.SurveillancePensee()
        s.pousser("Donc 6 x 7 = 42, puis je continue. ")
        assert s.alerte is None

    def test_la_phrase_nait_quand_elle_est_finie(self):
        """Une egalite coupee en deux ne doit pas etre jugee sur sa moitie."""
        s = ac.SurveillancePensee()
        s.pousser("Le produit vaut 6 x 7 = ")
        assert s.alerte is None            # « = » sans membre droit : rien
        s.pousser("41.")
        assert s.alerte is not None

    def test_une_seule_alerte_la_premiere(self):
        s = ac.SurveillancePensee()
        s.pousser("3 + 3 = 7. Et 4 + 4 = 9.")
        assert s.alerte is not None and "6" in s.alerte

    def test_pas_d_alerte_sur_une_pensee_juste(self):
        s = ac.SurveillancePensee()
        s.pousser("3 + 3 = 6. 4 + 4 = 8. Donc la somme est 14.")
        assert s.terminer() is None

    def test_derniere_phrase_sans_point_final(self):
        s = ac.SurveillancePensee()
        s.pousser("Et donc 3 + 3 = 7")
        assert s.terminer() is not None

    def test_interrupteur_coupe(self, monkeypatch):
        monkeypatch.setenv("AURA_AUTOCORRECTION", "0")
        s = ac.SurveillancePensee()
        assert s.actif is False
        s.pousser("2 + 2 = 5.")
        assert s.terminer() is None

    def test_reinitialiser_efface_l_alerte(self):
        """Sans ca, la correction se saboterait elle-meme : l'alerte de la
        PENSEE 1 couperait aussi la generation corrigee."""
        s = ac.SurveillancePensee()
        s.pousser("2 + 2 = 5.")
        assert s.alerte is not None
        s.reinitialiser()
        assert s.alerte is None
        s.pousser("2 + 2 = 4.")
        assert s.terminer() is None

    def test_inactif_ne_surveille_rien(self):
        s = ac.SurveillancePensee(actif_=False)
        s.pousser("2 + 2 = 5.")
        assert s.terminer() is None


class TestNoteInjectee:
    def test_la_note_est_adressee_au_modele(self):
        note = ac.note_pour_prompt("2 + 2 ne font pas 5 mais 4")
        assert "VERIFICATION" in note and "2 + 2" in note
        # elle demande de CORRIGER, elle ne dicte pas la reponse
        assert "reponse" in note.lower()


@pytest.mark.parametrize("texte,doit_parler", [
    ("1 + 1 = 2", False),
    ("1 + 1 = 3", True),
    ("le total est 10", False),
    ("100 - 1 = 98", True),
    ("100 - 1 = 99", False),
])
def test_table_de_verite(texte, doit_parler):
    assert (ac.verifier(texte) is not None) is doit_parler
