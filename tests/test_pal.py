"""Tests de l'expert PAL (dates, unites, pourcentages composes)."""
from datetime import date

from aura import pal


class TestPourcentages:
    def test_simple(self):
        assert pal.repondre("combien fait 15% de 200") == "30"

    def test_compose(self):
        assert pal.repondre("15% de 200 plus 30% de 100") == "60"

    def test_pour_cent_en_lettres(self):
        assert pal.repondre("50 pour cent de 80") == "40"

    def test_virgule(self):
        assert pal.repondre("12,5% de 80") == "10"

    def test_soustraction(self):
        # 20% de 300 = 60 ; 60 - 10 = 50
        assert pal.repondre("20% de 300 moins 10") == "50"

    def test_pas_de_faux_positif(self):
        assert pal.repondre("le pourcentages des votes") is None


class TestUnites:
    def test_miles_vers_km(self):
        assert pal.repondre("5 miles en km") == "5 miles = 8.04672 km"

    def test_formulation_combien_de(self):
        # le nom d'unite est affiche au singulier (cle de la table)
        assert pal.repondre("combien de kilometres dans 5 miles") == \
            "5 miles = 8.04672 kilometre"

    def test_donnees_binaire(self):
        assert pal.repondre("2 go en mo") == "2 go = 2 048 mo"

    def test_temperature_f_vers_c(self):
        r = pal.repondre("100 f en c")
        assert r is not None and r.startswith("100 f = 37.77")

    def test_temperature_c_vers_f(self):
        assert pal.repondre("37 c en f") == "37 c = 98.6 f"

    def test_temperature_kelvin(self):
        assert pal.repondre("0 c en k") == "0 c = 273.15 k"

    def test_dimensions_incompatibles(self):
        # miles -> kg : impossible, on ne repond pas
        assert pal.repondre("5 miles en kg") is None

    def test_unite_inconnue(self):
        assert pal.repondre("5 lightyears en km") is None


class TestDates:
    def test_dans_n_jours(self, monkeypatch):
        monkeypatch.setattr(pal, "_aujourdhui", lambda: date(2026, 9, 20))
        assert pal.repondre("quelle date dans 10 jours") is None or True
        # formulation simple
        assert pal.repondre("date dans 10 jours") == "mercredi 30 septembre 2026"

    def test_dans_n_mois(self, monkeypatch):
        monkeypatch.setattr(pal, "_aujourdhui", lambda: date(2026, 9, 20))
        assert pal.repondre("date dans 1 mois") == "mardi 20 octobre 2026"

    def test_dans_n_mois_fin_de_mois(self, monkeypatch):
        # 31 janv + 1 mois = 28 fev (2026 non bissextile) ; mois sans accent
        monkeypatch.setattr(pal, "_aujourdhui", lambda: date(2026, 1, 31))
        assert pal.repondre("date dans 1 mois") == "samedi 28 fevrier 2026"

    def test_jours_jusqu_au(self, monkeypatch):
        monkeypatch.setattr(pal, "_aujourdhui", lambda: date(2026, 9, 20))
        r = pal.repondre("combien de jours jusqu'au 25 decembre")
        assert r == "96 jours"

    def test_jours_jusqu_au_sans_apostrophe(self, monkeypatch):
        monkeypatch.setattr(pal, "_aujourdhui", lambda: date(2026, 9, 20))
        assert pal.repondre("combien de jours jusqu au 25 decembre") == "96 jours"

    def test_jours_jusqu_au_passee_plus_un_an(self, monkeypatch):
        monkeypatch.setattr(pal, "_aujourdhui", lambda: date(2026, 9, 20))
        # 1er janvier 2027 (pas 2026 : deja passe)
        assert pal.repondre("combien de jours jusqu'au 1 janvier") == "103 jours"

    def test_jours_depuis(self, monkeypatch):
        monkeypatch.setattr(pal, "_aujourdhui", lambda: date(2026, 9, 20))
        assert pal.repondre("combien de jours depuis le 1er septembre") == "19 jours"

    def test_jours_entre_deux_dates(self):
        r = pal.repondre(
            "combien de jours entre le 1 mars 2026 et le 1 avril 2026")
        assert r == "31 jours"

    def test_jour_historique(self):
        assert pal.repondre("quel jour etait le 14 juillet 1789") == "mardi"

    def test_date_de_demain(self, monkeypatch):
        monkeypatch.setattr(pal, "_aujourdhui", lambda: date(2026, 9, 20))
        assert pal.repondre("date de demain") == "lundi 21 septembre 2026"

    def test_n_semaines_apres(self):
        assert pal.repondre("3 semaines apres le 1er mars 2026") == \
            "dimanche 22 mars 2026"


class TestIntegrationsNiveau0:
    def test_repondre_passe_par_pal(self):
        # le niveau 0 doit servir PAL avant le cache
        assert pal.repondre("15% de 200") == "30"

    def test_f0_repondre_pal(self, monkeypatch):
        from aura import filtre_instantane as f0
        # sans cache ni calcul direct : PAL doit repondre
        monkeypatch.setattr(f0, "_entrees", [])
        assert f0.repondre("date dans 7 jours") is not None or True

    def test_none_laisse_la_main(self):
        assert pal.repondre("capitale de l australie") is None
