"""Expert encyclopedique (Wikipedia) et reponses horaires : reseau simule."""
from datetime import datetime

import pytest

from aura import encyclopedie
from aura import orchestrateur


def _faux_reseau(monkeypatch, recherches: dict, extraits: dict):
    """recherches : requete -> (titres, suggestion) ; extraits : titre -> page."""
    def _json(url: str):
        if "list=search" in url:
            requete = url.split("srsearch=")[1].split("&")[0]
            requete = requete.replace("+", " ").replace("%27", "'")
            titres, suggestion = recherches.get(requete, ([], None))
            info = {"suggestion": suggestion} if suggestion else {}
            return {"query": {"search": [{"title": t} for t in titres],
                              "searchinfo": info}}
        titre = url.split("titles=")[1].split("&")[0].replace("+", " ")
        page = extraits.get(titre)
        return {"query": {"pages": [page or {"missing": True}]}}

    monkeypatch.setattr(encyclopedie.apis_publiques, "_json", _json)


@pytest.mark.parametrize("question, attendu", [
    ("qui est Elon Musk ?", "elon musk"),
    ("Qui est Steave Jobs", "steave jobs"),
    ("c'est quoi la photosynthèse", "photosynthese"),
    ("qu'est-ce que la gravitation", "gravitation"),
    ("qui est le président de la France", None),   # temporel : pas d'encyclo
    ("qui es tu", None),                           # pronom
    ("qui est aura", None),                        # nom de l'assistant
    ("c'est quoi ça", None),                       # pronom demonstratif
    ("qui est Napoleon 1er", None),                # chiffre : on s'abstient
    ("donne un code python", None),                # pas une definition
])
def test_sujet_de(question, attendu):
    assert encyclopedie.sujet_de(question) == attendu


def test_proche_tolere_faute_mais_pas_homonyme():
    assert encyclopedie._proche("steave jobs", "Steve Jobs")
    assert encyclopedie._proche("elon musk", "Elon Musk")
    assert encyclopedie._proche("mercure", "Mercure (planète)")
    assert not encyclopedie._proche("steave jobs", "Nate Landwehr")


def test_repond_avec_correction_et_source(monkeypatch):
    monkeypatch.setenv("AURA_APIS", "1")   # réactiver les API pour ce test
    _faux_reseau(
        monkeypatch,
        recherches={"steave jobs": (["Nate Landwehr"], "steve jobs"),
                    "steve jobs": (["Steve Jobs"], None)},
        extraits={"Steve Jobs": {
            "title": "Steve Jobs",
            "fullurl": "https://fr.wikipedia.org/wiki/Steve_Jobs",
            "extract": "Steven Paul Jobs, né en 1955, est un entrepreneur "
                       "américain cofondateur d'Apple."}})
    reponse = encyclopedie.repondre("qui est steave jobs")
    assert reponse is not None
    assert "cofondateur d'Apple" in reponse
    assert "[Wikipédia — Steve Jobs](https://fr.wikipedia.org/wiki/Steve_Jobs)" in reponse


def test_s_abstient_si_aucun_titre_proche(monkeypatch):
    _faux_reseau(monkeypatch,
                 recherches={"zorblax quimp": (["Nate Landwehr"], None)},
                 extraits={})
    assert encyclopedie.repondre("qui est zorblax quimp") is None


def test_s_abstient_sur_une_homonymie(monkeypatch):
    _faux_reseau(
        monkeypatch,
        recherches={"mercure": (["Mercure"], None)},
        extraits={"Mercure": {
            "title": "Mercure", "fullurl": "https://x", "pageprops":
            {"disambiguation": ""},
            "extract": "Mercure peut désigner plusieurs choses, dont la planète."}})
    assert encyclopedie.repondre("qui est mercure") is None


def test_desactivable(monkeypatch):
    monkeypatch.setenv("AURA_ENCYCLOPEDIE", "0")
    assert encyclopedie.repondre("qui est Elon Musk") is None


@pytest.mark.parametrize("question, attendu", [
    ("quelle heure est-il ?", "Il est 14 h 05."),
    ("Quelle heure est il", "Il est 14 h 05."),
    ("quel jour sommes-nous", "Nous sommes le mardi 13 octobre 2026."),
    ("quelle est la date d'aujourd'hui", "Nous sommes le mardi 13 octobre 2026."),
    ("quelle heure de train pour Lyon ?", None),
    ("quelle est la date de naissance de Victor Hugo", None),
])
def test_reponse_temps(question, attendu):
    maintenant = datetime(2026, 10, 13, 14, 5)
    assert orchestrateur._reponse_temps(question, maintenant) == attendu
