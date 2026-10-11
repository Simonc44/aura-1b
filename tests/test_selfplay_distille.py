"""Tests du distillateur self-play (scripts/selfplay_distille.py).

On ne teste QUE les fonctions pures : l'evaluation exacte des calculs et la
lecture des reponses du modele. La boucle de self-play elle-meme appelle le
cerveau local (lent) et n'a pas sa place dans une suite unitaire.

Enjeu reel : c'est ce filtre qui decide quelles paires partent en
entrainement LoRA. Un filtre trop laxiste distille du bruit — donc on teste
les deux sens, ce qui doit PASSER et ce qui doit etre JETE.
"""
import importlib
from fractions import Fraction

sp = importlib.import_module("scripts.selfplay_distille")


# ── evaluation exacte ───────────────────────────────────────────────────

def test_resoudre_entiers_et_decimaux():
    assert sp._resoudre("45 * 1.2") == Fraction(54)
    assert sp._resoudre("(2 + 3) * 4") == Fraction(20)
    assert sp._resoudre("10 / 4") == Fraction(5, 2)
    assert sp._resoudre("2 ** 10") == Fraction(1024)
    assert sp._resoudre("-7 + 2") == Fraction(-5)


def test_resoudre_exactitude_fracionnaire():
    """1/3 doit rester exact : un float ferait echouer la comparaison."""
    assert sp._resoudre("1 / 3") == Fraction(1, 3)
    assert sp._resoudre("1 / 3") * 3 == 1


def test_resoudre_refuse_le_code_et_l_absurde():
    """Aucune expression produit par un modele ne doit pouvoir executer du code."""
    assert sp._resoudre("__import__('os').system('echo hi')") is None
    assert sp._resoudre("open('x')") is None
    assert sp._resoudre("a + b") is None
    assert sp._resoudre("1 + ") is None
    assert sp._resoudre("") is None
    assert sp._resoudre(None) is None
    assert sp._resoudre("2 ** 100000") is None      # deni de service
    assert sp._resoudre("1 / 0") is None
    assert sp._resoudre("9" * 300) is None          # longueur bornee


def test_resoudre_puissance_non_entiere_refusee():
    assert sp._resoudre("2 ** 0.5") is None


# ── lecture du format du piegeur ────────────────────────────────────────

_PIEGE = ("ENONCE: Un pull coute 45 euros, il augmente de 20%.\n"
          "CALCUL: 45 * 1.2\n"
          "REPONSE: 54\n")


def test_champs_lit_les_trois_lignes():
    champs = sp._champs(_PIEGE)
    assert champs["ENONCE"].startswith("Un pull")
    assert champs["CALCUL"] == "45 * 1.2"
    assert champs["REPONSE"] == "54"


def test_champs_ignore_le_bruit_autour():
    bruit = "Voici mon piege :\n" + _PIEGE + "\nBonne chance !\n"
    champs = sp._champs(bruit)
    assert champs["CALCUL"] == "45 * 1.2"
    assert champs["REPONSE"] == "54"


def test_nombre_final_prend_le_dernier():
    assert sp._nombre_final("je calcule 3 puis 2, la reponse est 45") == 45
    assert sp._nombre_final("resultat : 6,5") == Fraction(13, 2)
    assert sp._nombre_final("aucun nombre ici") is None
    assert sp._nombre_final("") is None


def test_equivalent():
    assert sp._equivalent(Fraction(54), Fraction(54))
    assert not sp._equivalent(Fraction(54), Fraction(55))
    assert not sp._equivalent(None, Fraction(54))
    assert not sp._equivalent(Fraction(54), None)


def test_joli():
    assert sp._joli(Fraction(54)) == "54"
    assert sp._joli(Fraction(13, 2)) == "6.5"


# ── le filtre complet : cohérence interne du piegeur ────────────────────

def test_filtre_accepte_un_piege_coherent():
    champs = sp._champs(_PIEGE)
    cible = sp._resoudre(champs["CALCUL"])
    annonce = sp._nombre_final(champs["REPONSE"])
    assert sp._equivalent(cible, annonce)


def test_filtre_rejette_un_piege_incoherent():
    """Le modele annonce 55 alors que son propre calcul vaut 54 : jete."""
    faux = ("ENONCE: Un pull a 45 euros augmente de 20%.\n"
            "CALCUL: 45 * 1.2\n"
            "REPONSE: 55\n")
    champs = sp._champs(faux)
    cible = sp._resoudre(champs["CALCUL"])
    annonce = sp._nombre_final(champs["REPONSE"])
    assert not sp._equivalent(cible, annonce)


def test_filtre_rejette_un_calcul_inverifiable():
    """Sans CALCUL exploitable, la cible est inconnue : jete."""
    champs = sp._champs("ENONCE: Combien de pommes ?\nREPONSE: 3\n")
    assert sp._resoudre(champs.get("CALCUL", "")) is None
