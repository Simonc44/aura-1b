"""Micro-faits : le savoir elementaire chiffre que le 1B rate encore.

Pourquoi ce tier existe
-----------------------
Mesure au sous-banc (`scripts/sous_banc.py`, reference du 7 oct. 2026) : quatre
questions du banc 100 tombaient sur `llama-3.2-1b` alors qu'un enfant y repond
juste, et instantanement :

  « Combien de jours compte une annee bissextile ? »  -> 366
  « Combien de cotes a un hexagone ? »                -> 6
  « De quoi les plantes ont-elles besoin pour la photosynthese ? »
  « Donne trois couleurs primaires separees par des virgules. »

Ce n'est PAS une limite de raisonnement (ces questions n'ont pas d'etapes) ni
de vocabulaire : c'est une limite de RAPPEL. Un modele de 1 milliard de
parametres, quantifie en 4 bits, ne stocke pas de facon fiable qu'une annee
bissextile a 366 jours ; il redige une phrase plausible et se trompe. Un
programme, lui, ne se trompe pas — et coute moins d'une milliseconde.

Ce que ce tier n'est PAS
------------------------
Une encyclopedie. La table est COURTE et volontairement limitee aux comptages
stables et non ambigus (durees, cotes des polygones, couleurs primaires,
photosynthese). Le savoir etendu vit dans `rag-binaire`, verifie et relu ; le
tier-ci ne doit surtout pas court-circuiter une question a laquelle il ne sait
pas repondre — d'ou l'abstention stricte : toute forme qui n'est pas
exactement celle d'une regle rend None, et la cascade continue.

Discipline identique a `calcul_achat` / `problemes_arithmetiques` : on ne
devine jamais. Un solveur qui devine est pire qu'un LLM, parce qu'il a l'air
d'avoir raison.
"""
from __future__ import annotations

import re
from fractions import Fraction

__all__ = ["resoudre", "COTES_POLYGONES"]


def _sans_accents(texte: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", texte.lower())
    sans = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sans).strip()


# ── Polygones ─────────────────────────────────────────────────────────────
# Seuls les noms NON ambigus sont listes : « cercle » n'a pas de cote, et un
# cube a des aretes, pas des cotes — les inclure serait deviner.
COTES_POLYGONES: dict[str, int] = {
    "triangle": 3,
    "quadrilatere": 4,
    "carre": 4,
    "rectangle": 4,
    "losange": 4,
    "pentagone": 5,
    "hexagone": 6,
    "heptagone": 7,
    "octogone": 8,
    "enneagone": 9,
    "nonagone": 9,
    "decagone": 10,
    "dodecagone": 12,
}

# La question doit PARLER de cotes : sans ce mot, une simple mention d'une
# figure (« dessine un hexagone ») ne doit pas declencher une reponse.
_RE_COTES = re.compile(r"\bcotes?\b")


# ── Durees et comptages ───────────────────────────────────────────────────
# Regles ordonnees : la plus specifique d'abord. « annee bissextile » doit
# passer AVANT « annee », sinon une annee bissextile vaudrait 365 jours.
_RE_BISSEXTILE = (r"\bbissextile\b", r"\bjours?\b")
_RE_SEMAINE = (r"\bsemaines?\b", r"\bjours?\b")
_RE_MOIS = (r"\bmois\b", r"\bannees?\b")
_RE_MINUTES = (r"\bminutes?\b", r"\bheures?\b")
_RE_SECONDES = (r"\bsecondes?\b", r"\bminutes?\b")
_RE_HEURES = (r"\bheures?\b", r"\bjournees?\b|\bjours?\b")
_RE_ANNEE = (r"\bannees?\b", r"\bjours?\b")
_RE_NOMBRE = re.compile(r"\d+(?:[.,]\d+)?")

# Conversions de duree a resultat exact. La quantite doit etre ecrite avec
# son unite source ("3 jours") et etre le seul nombre de la question :
# sinon on s'abstient plutot que de choisir une interpretation.
_CONVERSIONS_DUREE = (
    (r"\bjours?\b", r"\bannees?\s+bissextiles?\b", 366, "jours"),
    (r"\bjours?\b", r"\bsemaines?\b", 7, "jours"),
    (r"\bmois\b", r"\bannees?\b", 12, "mois"),
    (r"\bminutes?\b", r"\bheures?\b", 60, "minutes"),
    (r"\bsecondes?\b", r"\bminutes?\b", 60, "secondes"),
    (r"\bheures?\b", r"\b(?:journees?|jours?)\b", 24, "heures"),
    (r"\bjours?\b", r"\bannees?\b", 365, "jours"),
)


def _compte(q: str, motifs: tuple[str, ...]) -> bool:
    """La question porte-t-elle sur TOUS ces motifs (et parle d'un nombre) ?"""
    return all(re.search(m, q) for m in motifs)


# Une question de comptage : « combien », « nombre de », « quel est le nombre ».
_RE_COMBien = re.compile(r"\bcombien\b|\bnombre\s+de\b|\bquel\s+est\s+le\s+nombre\b")


# ── Regles a reponse directe ──────────────────────────────────────────────

def _polygone(q: str) -> str | None:
    if not _RE_COTES.search(q) or not _RE_COMBien.search(q):
        return None
    trouves = [(nom, cotes) for nom, cotes in COTES_POLYGONES.items()
               if re.search(rf"\b{nom}s?\b", q)]
    if len(trouves) != 1:
        return None
    nom, cotes = trouves[0]
    return f"Un {nom} a {cotes} cotes."


def _duree(q: str) -> str | None:
    if not _RE_COMBien.search(q):
        return None
    nombres = _RE_NOMBRE.findall(q)
    if len(nombres) == 1:
        for cible, source, facteur, libelle in _CONVERSIONS_DUREE:
            if not re.search(cible, q):
                continue
            quantite = re.search(
                r"(\d+(?:[.,]\d+)?)\s*" + source, q)
            if quantite:
                valeur = Fraction(quantite.group(1).replace(",", ".")) * facteur
                rendu = (str(valeur.numerator) if valeur.denominator == 1
                         else f"{float(valeur):.6g}")
                return f"{rendu} {libelle}."
    if nombres:
        # Une quantite est presente mais aucune conversion univoque ne
        # correspond : ne pas appliquer ensuite une constante generique.
        return None
    if _compte(q, _RE_BISSEXTILE):
        return "Une annee bissextile compte 366 jours."
    if _compte(q, _RE_SEMAINE):
        return "Une semaine compte 7 jours."
    if _compte(q, _RE_MOIS):
        return "Une annee compte 12 mois."
    if _compte(q, _RE_MINUTES):
        return "Une heure compte 60 minutes."
    if _compte(q, _RE_SECONDES):
        return "Une minute compte 60 secondes."
    if _compte(q, _RE_HEURES):
        return "Une journee compte 24 heures."
    if _compte(q, _RE_ANNEE):
        return "Une annee ordinaire compte 365 jours."
    return None


# « couleur(s) primaire(s) » : la question nomme la notion, on peut repondre.
_RE_PRIMAIRES = re.compile(r"\bcouleurs?\s+primaires?\b")
_RE_INTENTION_COULEURS = re.compile(
    r"\b(quelles?|combien|donne[rz]?|cite[rz]?|nomme[rz]?|liste|"
    r"identifie[rz]?|sont)\b")


def _couleurs(q: str) -> str | None:
    if (not _RE_PRIMAIRES.search(q)
            or not _RE_INTENTION_COULEURS.search(q)):
        return None
    # Synthese soustractive : celles qu'on apprend a l'ecole (et que le banc
    # attend). On nomme AUSSI la variante additive pour rester exact.
    return ("Les trois couleurs primaires sont le rouge, le bleu et le jaune "
            "(en synthese additive : rouge, vert, bleu).")


_RE_PHOTOSYNTHESE = re.compile(r"\bphotosyntheses?\b")
_RE_INTENTION_BESOINS = re.compile(
    r"\b(besoins?|necessaire|necessaires|faut|requis|requise|"
    r"ingredients?|intrants?)\b")


def _photosynthese(q: str) -> str | None:
    if (not _RE_PHOTOSYNTHESE.search(q)
            or not _RE_INTENTION_BESOINS.search(q)):
        return None
    return ("Pour la photosynthese, les plantes ont besoin de lumiere (du "
            "Soleil), d'eau et de dioxyde de carbone (CO2), grace a la "
            "chlorophylle.")


# L'ordre des regles n'a pas d'importance : elles portent sur des mots
# distincts (cotes / durees / couleurs / photosynthese), jamais sur les memes.
_REGLES = (_polygone, _duree, _couleurs, _photosynthese)


def resoudre(question: str) -> str | None:
    """Repond a un micro-fait elementaire, ou None si la question n'en est pas un.

    Toujours synchrone, sans le moindre appel au modele ni au disque.
    """
    if not question:
        return None
    q = _sans_accents(question)
    for regle in _REGLES:
        reponse = regle(q)
        if reponse is not None:
            return reponse
    return None
