"""Titulaire d'une fonction (aura/titulaires.py) — SANS reseau.

Les reponses Wikidata sont simulees : on teste la LOGIQUE (forme de la
question, correspondance exacte, mandat en cours, abstentions), pas le
reseau. Cas reel a l'origine du module : « qui est le président de la France »
rendait un extrait web generique sans le nom.
"""
import urllib.parse

import pytest

from aura import apis_publiques, titulaires


def _mandat(qid, debut=None, fin=None, rang="normal", precision=11):
    qualif = {}
    if debut:
        qualif["P580"] = [{"datavalue": {"value": {
            "time": debut, "precision": precision}}}]
    if fin:
        qualif["P582"] = [{"datavalue": {"value": {
            "time": fin, "precision": 11}}}]
    return {"mainsnak": {"snaktype": "value", "datavalue": {
        "value": {"entity-type": "item", "id": qid}}},
        "rank": rang, "qualifiers": qualif}


class FauxWikidata:
    """Repond aux trois actions utilisees, comme le ferait l'API."""

    def __init__(self, recherches=None, mandats=None, libelles=None):
        self.recherches = recherches or {}      # terme -> liste de resultats
        self.mandats = mandats or {}            # (qid, prop) -> statements
        self.libelles = libelles or {}          # qid -> libelle
        self.appels: list[str] = []

    def __call__(self, url):
        self.appels.append(url)
        p = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        if p["action"] == "wbsearchentities":
            return {"search": self.recherches.get(p["search"], [])}
        if p["action"] == "wbgetclaims":
            cle = (p["entity"], p["property"])
            return {"claims": {p["property"]: self.mandats[cle]}
                    if cle in self.mandats else {}}
        if p["action"] == "wbgetentities":
            lib = self.libelles.get(p["ids"])
            return {"entities": {p["ids"]: {"labels": (
                {"fr": {"language": "fr", "value": lib}} if lib else {})}}}
        raise AssertionError(url)


def _installer(monkeypatch, faux):
    monkeypatch.setattr(apis_publiques, "_json", faux)
    monkeypatch.setattr(apis_publiques, "_actives", lambda: True)
    monkeypatch.setattr(titulaires, "_aujourdhui", lambda: (2026, 10, 11))


def _president_de_la_france():
    return FauxWikidata(
        recherches={"président de la France": [
            {"id": "Q999", "label": "Président du Sénat",
             "match": {"type": "label", "text": "Président du Sénat"}},
            {"id": "Q191954", "label": "Président de la République française",
             "match": {"type": "alias", "text": "président de la France"}},
        ]},
        mandats={("Q191954", "P1308"): [
            _mandat("Q329", "+2012-05-15T00:00:00Z", "+2017-05-14T00:00:00Z"),
            _mandat("Q3052772", "+2017-05-14T00:00:00Z", rang="preferred"),
        ]},
        libelles={"Q3052772": "Emmanuel Macron"})


class TestAnalyse:
    @pytest.mark.parametrize("question, syntagme", [
        ("qui est le président de la France", "le président de la France"),
        ("Qui est le président de la France ?", "le président de la France"),
        ("qui est actuellement le premier ministre du Canada ?",
         "le premier ministre du Canada"),
        ("c'est qui le roi d'Espagne", "le roi d'Espagne"),
        ("Qui est la reine du Danemark", "la reine du Danemark"),
        ("qui est le maire de Paris aujourd'hui ?", "le maire de Paris"),
        ("Qui est la chancelière d'Allemagne ?", "la chancelière d'Allemagne"),
    ])
    def test_formes_reconnues(self, question, syntagme):
        assert titulaires.analyser(question)["syntagme"] == syntagme

    @pytest.mark.parametrize("question", [
        "Qui est Emmanuel Macron ?",                     # pas une fonction
        "qui était le président de la France en 1995",   # passe / historique
        "qui est le président de la France en 2012",     # annee = historique
        "qui est le président de la France et de l'Italie",   # compose
        "Quelle est la capitale de la France ?",
        "qui est le directeur de la NASA",               # fonction non couverte
        "Combien font 2+2 ?",
    ])
    def test_abstention_de_forme(self, question):
        assert titulaires.analyser(question) is None


class TestResoudre:
    def test_cas_reel_le_nom_est_donne(self, monkeypatch):
        faux = _president_de_la_france()
        _installer(monkeypatch, faux)
        rep = titulaires.resoudre("qui est le président de la France")
        assert rep.startswith("Le président de la France est Emmanuel Macron")
        assert "depuis le 14 mai 2017" in rep
        assert "[Wikidata](https://www.wikidata.org/wiki/Q191954)" in rep
        assert titulaires.est_reponse_titulaire(rep)

    def test_une_correspondance_approximative_est_refusee(self, monkeypatch):
        """Seul « Président du Sénat » existe : jamais sa reponse pour la France."""
        faux = FauxWikidata(
            recherches={"président de la France": [
                {"id": "Q999", "label": "Président du Sénat",
                 "match": {"type": "label", "text": "Président du Sénat"}}]},
            mandats={("Q999", "P1308"): [_mandat("Q1", "+2020-01-01T00:00:00Z")]},
            libelles={"Q1": "Quelqu'un D'autre"})
        _installer(monkeypatch, faux)
        assert titulaires.resoudre("qui est le président de la France") is None

    def test_repli_sur_le_pays_P35(self, monkeypatch):
        faux = FauxWikidata(
            recherches={
                "président de la France": [],
                "France": [
                    {"id": "Q1", "label": "France", "description": "film de 2021",
                     "match": {"type": "label", "text": "France"}},
                    {"id": "Q142", "label": "France",
                     "description": "pays d'Europe de l'Ouest",
                     "match": {"type": "label", "text": "France"}}]},
            mandats={("Q142", "P35"): [
                _mandat("Q329", "+2012-05-15T00:00:00Z", "+2017-05-14T00:00:00Z"),
                _mandat("Q3052772", "+2017-05-14T00:00:00Z")]},
            libelles={"Q3052772": "Emmanuel Macron"})
        _installer(monkeypatch, faux)
        rep = titulaires.resoudre("qui est le président de la France")
        assert "Emmanuel Macron" in rep and "Q142" in rep   # le pays, pas le film

    def test_premier_ministre_utilise_P6(self, monkeypatch):
        faux = FauxWikidata(
            recherches={
                "premier ministre du Canada": [],
                "Canada": [{"id": "Q16", "label": "Canada",
                            "description": "pays d'Amérique du Nord",
                            "match": {"type": "label", "text": "Canada"}}]},
            mandats={("Q16", "P6"): [_mandat("Q5", "+2025-03-14T00:00:00Z",
                                             precision=9)]},
            libelles={"Q5": "Un Premier Ministre"})
        _installer(monkeypatch, faux)
        rep = titulaires.resoudre("qui est le premier ministre du Canada ?")
        assert rep.startswith("Le premier ministre du Canada est Un Premier")
        assert "depuis 2025" in rep

    def test_deux_fonctions_deux_personnes_on_s_abstient_de_choisir(
            self, monkeypatch):
        """Deux postes homonymes aux titulaires differents : pas de pari."""
        faux = FauxWikidata(
            recherches={"président de la France": [
                {"id": "QA", "match": {"type": "alias",
                                       "text": "président de la France"}},
                {"id": "QB", "match": {"type": "alias",
                                       "text": "Président de la France"}}]},
            mandats={("QA", "P1308"): [_mandat("Q10", "+2017-01-01T00:00:00Z")],
                     ("QB", "P1308"): [_mandat("Q20", "+2018-01-01T00:00:00Z")]},
            libelles={"Q10": "A", "Q20": "B"})
        _installer(monkeypatch, faux)
        # pas de lieu connu non plus : abstention totale
        assert titulaires.resoudre("qui est le président de la France") is None

    def test_tous_les_mandats_termines_abstention(self, monkeypatch):
        faux = FauxWikidata(
            recherches={"roi d'Espagne": [
                {"id": "Q7", "match": {"type": "label", "text": "roi d'Espagne"}}]},
            mandats={("Q7", "P1308"): [
                _mandat("Q1", "+1975-11-22T00:00:00Z", "+2014-06-18T00:00:00Z")]},
            libelles={"Q1": "Juan Carlos"})
        _installer(monkeypatch, faux)
        assert titulaires.resoudre("qui est le roi d'Espagne") is None

    def test_une_fin_future_n_est_pas_une_fin(self, monkeypatch):
        faux = FauxWikidata(
            recherches={"président de la France": [
                {"id": "Q191954",
                 "match": {"type": "alias", "text": "président de la France"}}]},
            mandats={("Q191954", "P1308"): [
                _mandat("Q3052772", "+2017-05-14T00:00:00Z",
                        "+2027-05-14T00:00:00Z")]},
            libelles={"Q3052772": "Emmanuel Macron"})
        _installer(monkeypatch, faux)
        assert "Emmanuel Macron" in titulaires.resoudre(
            "qui est le président de la France")

    def test_hors_ligne_abstention(self, monkeypatch):
        _installer(monkeypatch, lambda url: None)
        assert titulaires.resoudre("qui est le président de la France") is None

    def test_api_coupee_aucun_appel(self, monkeypatch):
        faux = _president_de_la_france()
        _installer(monkeypatch, faux)
        monkeypatch.setattr(apis_publiques, "_actives", lambda: False)
        assert titulaires.resoudre("qui est le président de la France") is None
        assert faux.appels == []

    def test_question_hors_sujet_aucun_appel(self, monkeypatch):
        faux = _president_de_la_france()
        _installer(monkeypatch, faux)
        assert titulaires.resoudre("Décris le vent du nord.") is None
        assert faux.appels == []

    def test_libelle_absent_abstention(self, monkeypatch):
        faux = _president_de_la_france()
        faux.libelles.clear()
        _installer(monkeypatch, faux)
        assert titulaires.resoudre("qui est le président de la France") is None


class TestDates:
    def test_date_fr(self):
        assert titulaires._date_fr((2017, 5, 14, 11)) == "le 14 mai 2017"
        assert titulaires._date_fr((2017, 5, 1, 11)) == "le 1er mai 2017"
        assert titulaires._date_fr((2017, 5, 0, 10)) == "mai 2017"
        assert titulaires._date_fr((2017, 0, 0, 9)) == "2017"
