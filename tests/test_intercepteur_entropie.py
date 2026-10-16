import math

from aura.intercepteur_entropie import marge_top2, premier_doute


def test_marge_top2_deux_candidats():
    lp = {"a": math.log(0.7), "b": math.log(0.2)}
    assert abs(marge_top2(lp) - 0.5) < 1e-9


def test_marge_top2_ordre_indifferent():
    lp = {"b": math.log(0.2), "a": math.log(0.7), "c": math.log(0.05)}
    assert abs(marge_top2(lp) - 0.5) < 1e-9


def test_marge_top2_un_seul_candidat():
    assert marge_top2({"a": 0.0}) == 1.0


def test_marge_top2_vide_suppose_doute():
    assert marge_top2({}) == 0.0


def test_premier_doute_trouve_le_premier():
    assert premier_doute([0.9, 0.5, 0.1, 0.05], seuil=0.15) == 2


def test_premier_doute_aucun():
    assert premier_doute([0.9, 0.5], seuil=0.15) is None


def test_premier_doute_seuil_strict():
    # une marge egale au seuil n'est pas un doute
    assert premier_doute([0.15], seuil=0.15) is None
