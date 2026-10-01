"""Detecteur de concepts abstraits : la question -> le concept derriere.

Plutot que de laisser le LLM deviner de quoi parle l'utilisateur, la
question est vectorisee (~3 ms, aura/embeddings.py) et comparee aux
descriptions de la bibliotheque de concepts. Au-dessus du seuil, le
LEXIQUE du concept est injecte dans le prompt systeme
(orchestrateur._generer) : le 1B recoit les termes justes au lieu de
les inventer.

Bibliotheque calibree (micro-eval 12 paires requete->concept) :
- descriptions COURTES (1 phrase) avec une tournure d'exemple « ... » :
  7/12 bien classes avec des definitions longues -> 11/12 en court ;
- similarite cosinus hors-sujet ~0.15-0.30, concerne >= _SEUIL_DEFAUT ;
- matrice de la bibliotheque memoiree : ~3 ms par detection, ~500 ms
  seulement a la premiere (encodage des 30 descriptions).

AURA_CONCEPTS=0 coupe la detection ; modele absent = detection inerte
(embeddings.encoder -> None) : jamais d'erreur, comportement inchange.
"""
import logging
import os

import numpy as np

from . import embeddings

LOG = logging.getLogger("aura.concepts")

_SEUIL_DEFAUT = 0.40     # calibre : hors-sujet <= ~0.35, concerne >= 0.43
# Garde-fou de longueur (micro-eval 12 vrais / 5 faux) : les vrais
# concepts s expriment en >= 8 mots, les faux positifs (meteo, blague,
# calcul) sont tous <= 6 — une question courte ne contient pas assez de
# contexte pour affirmer un concept.
_MOTS_MIN = 7

# matrice de la bibliotheque memoiree (voir _vecteurs_concepts)
_cache_matrice: np.ndarray | None = None
_cache_cle: tuple | None = None
_cache_noms: list[str] | None = None

# nom -> description : le texte vectorise ET le lexique injecte
_CONCEPTS: dict[str, str] = {
    "angoisse-existentielle": (
        "angoisse existentielle, mauvaise foi : se sentir piege par ses "
        "propres choix, le vertige de la liberte, on est condamne a "
        "choisir (Kierkegaard, Sartre)."),
    "absurde-et-revolte": (
        "l absurde et la revolte : chercher du sens dans un monde qui "
        "n en a pas, coexister avec l absurdite sans renoncer (Camus, "
        "mythe de Sisyphe)."),
    "determinisme-liberte": (
        "determinisme contre libre arbitre : si tout est cause par les "
        "genes et l environnement, suis-je vraiment libre et "
        "responsable ? (compatibilisme)"),
    "utilitarisme": (
        "utilitarisme : le bien est le plus grand bonheur du plus grand "
        "nombre ; vaut-il mieux sauver cinq en sacrifiant un ? "
        "(Bentham, Mill)"),
    "imperatif-categorique": (
        "imperatif categorique : agis seulement selon la maxime que tu "
        "pourrais vouloir loi universelle ; le devoir et la dignite "
        "contre le calcul des consequences (Kant)."),
    "syllogisme": (
        "syllogisme : premisses universelles puis conclusion necessaire, "
        "« si tous les A sont B, et x est A, alors x est B » "
        "(validite contre veracite)."),
    "preuve-par-absurde": (
        "preuve par l absurde : supposer le contraire de la these, "
        "trouver une contradiction, en conclure que la these est vraie "
        "(mathematiques, logique)."),
    "causalite-vs-correlation": (
        "correlation n est pas causalite : deux varient ensemble sans "
        "lien de cause, facteur confondant, post hoc ergo propter hoc."),
    "biais-cognitifs": (
        "biais cognitif : biais de confirmation, je ne retiens que ce "
        "qui confirme mon opinion ; aversion a la perte, biais de "
        "survie, illusion statistique."),
    "sophismes": (
        "sophisme : faux dilemme, homme de paille, appel a l autorite, "
        "pente glissante, ad hominem — des arguments invalides qui "
        "semblent convaincants."),
    "pensee-systemique": (
        "pensee systemique : boucles de retroaction, je corrige ici et "
        "le probleme resurgit ailleurs (effet rebond, emergence, "
        "fluxe)."),
    "rasoir-d-occam": (
        "rasoir d Ockham : entre deux explications compatibles, "
        "preferer celle qui invente le moins d hypotheses "
        "(parcimonie)."),
    "charge-cognitive": (
        "charge cognitive : trop de choses a faire en meme temps, la "
        "memoire de travail deborde, sous stress on traite moins bien "
        "(attention, automatismes)."),
    "procrastination": (
        "procrastination : repousser une tache desagreable pour eviter "
        "la mauvaise humeur qu elle provoque, regret apres coup "
        "(regulation emotionnelle)."),
    "croissance-vs-fixed": (
        "etats d esprit : croire que l intelligence se developpe par "
        "l effort (croissante) contre la voir immuable (fixe) ; "
        "reaction a l echec (Dweck)."),
    "double-bind": (
        "double contrainte (double bind) : deux injonctions "
        "incompatibles ou toute reponse est faute, communication qui "
        "se contredit en meta."),
    "alignement-ia": (
        "alignement IA : specifier une fonction objectif qui capture la "
        "vraie intention ; Goodhart — optimiser la mesure plutot que "
        "la tache visee."),
    "biais-recompense": (
        "reward hacking : l agent triche la metrique au lieu de faire "
        "la tache, la recompense devient la cible (comportement de "
        "passe-passe)."),
    "metriques-de-vanite": (
        "metriques de vanite : choisir le chiffre qui monte plutot que "
        "le resultat reel, numbers go up, paradoxe de Goodhart, "
        "apparence de performance."),
    "identite-narrative": (
        "identite narrative : l histoire qu on se raconte pour rester "
        "le meme, on re ecrit le passe pour eviter la dissonance "
        "cognitive."),
    "zones-de-non-savoir": (
        "savoir qu on ne sait pas : cone de l ignorance, distinguer "
        "l incertain du probable, humilite epistemique, mesure de ce "
        "qu on ignore."),
    "effet-Matthieu": (
        "effet Matthieu : celui qui a deja recu davantage en recoit "
        "encore, les avantages s accumulent, cercle vertueux ou "
        "vicieux, inegalites qui se renforcent."),
    "anti-fragilite": (
        "anti-fragilite : ce qui gagne au desordre et aux chocs contre "
        "ce qui peine a survivre, optionnalite, sur-observer les "
        "protections (Taleb)."),
    "preuve-par-induction": (
        "raisonnement inductif : des cas passes vers une regle "
        "probable ; probleme de l induction de Hume, on surestime les "
        "regularites observees."),
    "explication-vs-description": (
        "expliquer contre decrire : COMMENT un phenomene se produit "
        "(mecanisme) et POURQUOI il a lieu (cause, sens) ; le fait "
        "contre la justification."),
    "niveau-d-abstraction": (
        "niveaux d abstraction : des mecanismes de bas niveau emergent "
        "des proprietes de haut niveau ; reductionnisme contre "
        "realisme des niveaux."),
    "heuristiques-decisions": (
        "heuristiques : regles rapides economes en cognition, "
        "satisfaisant contre optimisant, fidelite au reel plus que "
        "precision parfaite (Simon, Gigerenzer)."),
    "responsabilite": (
        "responsabilite : qui repond de quoi, faute contre risque, "
        "imputabilite d un systeme quand personne n a tout decide "
        "(individuel, collectif)."),
    "securite-vs-liberte": (
        "securite contre liberte : quelles libertes accepter de ceder "
        "pour se proteger, etat d urgence, la surveillance generalisee "
        "qui empeche ce qu elle promet."),
    "sens-du-travail": (
        "sens du travail : produire quelque chose de comprehensible et "
        "utile, maitrise et autonomie contre l alienation (Marx, "
        "sociologie du travail)."),
}


def actif() -> bool:
    """AURA_CONCEPTS != 0 (actif par defaut)."""
    return os.environ.get("AURA_CONCEPTS", "1") != "0"


def seuil() -> float:
    try:
        return float(os.environ.get("AURA_CONCEPTS_SEUIL",
                                    str(_SEUIL_DEFAUT)))
    except ValueError:
        return _SEUIL_DEFAUT


def _vecteurs_concepts() -> tuple[list[str], np.ndarray] | None:
    """(noms, matrice (n,384) normalisee) — None si indisponible.

    Matrice memoiree : encoder 30 descriptions coutait ~500 ms A CHAQUE
    detection ; on ne recalcule que si la bibliotheque change.
    """
    global _cache_matrice, _cache_cle, _cache_noms
    if not _CONCEPTS:
        return None
    cle = tuple(_CONCEPTS.items())
    if (_cache_matrice is not None and _cache_cle == cle
            and _cache_noms is not None):
        return _cache_noms, _cache_matrice
    noms = list(_CONCEPTS)
    vecs = embeddings.encoder([_CONCEPTS[n] for n in noms])
    if vecs is None:
        return None
    _cache_cle, _cache_noms, _cache_matrice = cle, noms, vecs
    return noms, vecs


def detecter(texte: str) -> dict | None:
    """Concept le plus proche si similarite >= seuil, sinon None.

    Renvoie {"concept": nom, "score": float, "description": texte}.
    """
    if not actif() or not (texte or "").strip():
        return None
    if len(texte.split()) < _MOTS_MIN:
        return None
    bibe = _vecteurs_concepts()
    if bibe is None:
        return None
    noms, matrice = bibe
    vec = embeddings.encoder1(texte)
    if vec is None:
        return None
    sims = matrice @ vec
    i = int(np.argmax(sims))
    score = float(sims[i])
    if score < seuil():
        return None
    return {"concept": noms[i], "score": round(score, 3),
            "description": _CONCEPTS[noms[i]]}


def ancrage(texte: str) -> str | None:
    """Bloc a injecter dans le prompt systeme, ou None.

    Court-circuit symbolique : le LLM recoit le lexique du concept des
    la premiere phrase, au lieu de partir de zero (ou d'inventer les
    termes).
    """
    hit = detecter(texte)
    if not hit:
        return None
    return (
        f"LEXIQUE (concept rapproche : {hit['concept']}, "
        f"similarite {hit['score']:.0%}) : {hit['description']}\n"
        "Si la question touche ce concept, appuie-toi sur ce lexique "
        "(termes justes, nuances) sans forcer le hors-sujet.")
