"""Ordre de la cascade : un EXACT ne doit jamais perdre contre un FLOU.

Defaut reel, corrige le 6 oct. 2026 : le cache semantique etait consulte par
le niveau 0, donc AVANT calcul_achat / calcul_verbal / calcul_taux /
tri-transitif / SAT / potcode. Une memoire approximative (cosinus MiniLM
>= 0,70 sur 384 dimensions) repondait donc a la place d'un calcul exact.

Consequence concrete : « on ne peut pas savoir ce que le systeme sait
CALCULER », puisque n'importe quelle reponse memorisee — y compris fausse —
pouvait court-circuiter le solveur. Observe en vrai : le cache a servi 240 a
une question de taux avant que `calcul_taux` ne la calcule.

Ces tests verrouillent les DEUX cotes de la propriete :
  * EMPOISONNE : le cache contient une reponse FAUSSE a une question que le
    solveur exact sait traiter -> le solveur exact doit gagner. Depuis le
    12 oct. 2026 une purge a la LECTURE retire en plus les entrees de CALCUL :
    le poison arithmetique ne peut plus exister (1re defense, testee dans
    TestEtagesSepares) ; la cascade ci-dessous verrouille la 2e ;
  * MUET : aucun solveur ne sait repondre -> le cache doit continuer a
    servir (sinon on aurait « corrige » en cassant la memoire).
"""
import json

import pytest

from aura import filtre_instantane as F
from aura.orchestrateur import RontoGPT

_Q_TAUX = ("Un train roule a 80 km/h pendant 3 heures. Quelle distance "
           "parcourt-il ?")
_Q_EXACT = "combien fait 45*12+3"
_Q_INCONNUE = "Quel est le nom de code du projet Zorblax ?"


def _cache(monkeypatch, tmp_path, entrees: list[dict]) -> None:
    """Installe un cache de test (jsonl) et force son rechargement."""
    fichier = tmp_path / "cache.jsonl"
    fichier.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entrees),
        encoding="utf-8")
    monkeypatch.setattr(F, "_FICHIER", fichier)
    monkeypatch.setattr(F, "_vecteurs", None)
    monkeypatch.setattr(F, "_vectoriseur", None)
    monkeypatch.setattr(F, "_matrice", None)
    monkeypatch.setattr(F, "_entrees", [])


@pytest.fixture
def _llm_muet(monkeypatch):
    """Aucun LLM : tout ce qui est genere renvoie une sentinelle."""
    from aura import llama_cerveau
    monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "[GENERE]")


class TestEtagesSepares:
    """`repondre` = exact, `repondre_cache` = flou : deux etages distincts."""

    def test_repondre_ignore_le_cache(self, monkeypatch, tmp_path):
        _cache(monkeypatch, tmp_path,
               [{"q": _Q_EXACT, "r": "999", "t": "2026-10-06"}])
        # Le poison d'un CALCUL ne dort meme plus dans le cache : il est retire
        # a la LECTURE (purge « fix bug 2 », 12 oct. 2026), et l'ordre exact
        # est la SECONDE defense. Le poison de l'etage flou se teste donc des
        #ormais sur un solveur exact NON arithmetique (test de cascade plus
        # bas, _Q_TAUX) : c'est la seule entree qui survit a la purge.
        assert F.repondre_cache(_Q_EXACT) is None
        # ...et pourtant l'etage exact repond juste, sans consulter le cache
        assert F.repondre(_Q_EXACT) == "543"
        # l'etage complet (bancs/outils) garde l'ancien comportement exact-first
        assert F.repondre_tout(_Q_EXACT) == "543"

    def test_non_mathematique_nest_pas_dans_repondre(self, monkeypatch,
                                                     tmp_path):
        _cache(monkeypatch, tmp_path,
               [{"q": _Q_INCONNUE, "r": "Zorblax-9", "t": "2026-10-06"}])
        assert F.repondre_cache(_Q_INCONNUE) == "Zorblax-9"
        assert F.repondre(_Q_INCONNUE) is None


class TestCascade:
    def test_cache_empoisonne_ne_bat_pas_un_solveur_exact(
            self, monkeypatch, tmp_path, _llm_muet):
        _cache(monkeypatch, tmp_path,
               [{"q": _Q_TAUX, "r": "160 km", "t": "2026-10-06"}])
        assert F.repondre_cache(_Q_TAUX) == "160 km"     # hit reel et faux

        r = RontoGPT().executer_detaille(_Q_TAUX)
        assert r["cerveau_choisi"] == "calcul-taux"
        assert r["reponse"] == "240"

    def test_cache_sert_quand_aucun_solveur_ne_sait(
            self, monkeypatch, tmp_path, _llm_muet):
        _cache(monkeypatch, tmp_path,
               [{"q": _Q_INCONNUE, "r": "Zorblax-9", "t": "2026-10-06"}])

        r = RontoGPT().executer_detaille(_Q_INCONNUE)
        assert r["cerveau_choisi"] == "cache-semantique"
        assert r["reponse"] == "Zorblax-9"
