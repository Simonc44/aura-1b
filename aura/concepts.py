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

BIBLIOTHEQUE EXTENSIBLE (concepts.toml) : la racine du projet peut
contenir un concepts.toml ([concepts] -> nom = "description", valeur
table avec description/domaine acceptee) qui complete ou SURCHARGE les
30 concepts integres (AURA_CONCEPTS_TOML pour un autre chemin).

REPLI DOMAIN (zéro muet) : sous le seuil mais au-dessus de
_SEUIL_INDICE, avec un mot-clé du domaine du concept le plus proche ->
un rappel GENERAL du domaine remplace le silence (AURA_CONCEPTS_REPLI=0
coupe).
"""
import logging
import os
import tomllib
import unicodedata
from pathlib import Path

import numpy as np

from . import embeddings

LOG = logging.getLogger("aura.concepts")

_SEUIL_DEFAUT = 0.40     # calibre : hors-sujet <= ~0.35, concerne >= 0.43
# Garde-fou de longueur (micro-eval 12 vrais / 5 faux) : les vrais
# concepts s expriment en >= 8 mots, les faux positifs (meteo, blague,
# calcul) sont tous <= 6 — une question courte ne contient pas assez de
# contexte pour affirmer un concept.
_MOTS_MIN = 7

# REPLI DOMAIN : bande d'indice entre le silence et le seuil (les scores
# reellement hors-sujet restent exclues : score ~0.35-0.40 SANS mot-cle
# de domaine du concept le plus proche -> zéro comme avant.
_SEUIL_INDICE = 0.30

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

# -- domaines (repli large, zéro muet) ----------------------------------

# concept -> domaine : sert au repli (bande d'indice) — un concept non
# affirmable mais PROCHE donne un rappel GENERAL de son domaine plutot
# que le silence. Les concepts de concepts.toml apportent le leur via la
# cle "domaine".
_DOMAINES: dict[str, str] = {
    "angoisse-existentielle": "philosophie",
    "absurde-et-revolte": "philosophie",
    "determinisme-liberte": "philosophie",
    "zones-de-non-savoir": "philosophie",
    "explication-vs-description": "philosophie",
    "niveau-d-abstraction": "philosophie",
    "rasoir-d-occam": "philosophie",
    "utilitarisme": "ethique",
    "imperatif-categorique": "ethique",
    "responsabilite": "ethique",
    "securite-vs-liberte": "ethique",
    "syllogisme": "logique",
    "preuve-par-absurde": "logique",
    "preuve-par-induction": "logique",
    "sophismes": "logique",
    "causalite-vs-correlation": "logique",
    "heuristiques-decisions": "logique",
    "biais-cognitifs": "psychologie",
    "charge-cognitive": "psychologie",
    "procrastination": "psychologie",
    "croissance-vs-fixed": "psychologie",
    "double-bind": "psychologie",
    "identite-narrative": "psychologie",
    "pensee-systemique": "sciences-sociales",
    "effet-Matthieu": "sciences-sociales",
    "anti-fragilite": "sciences-sociales",
    "sens-du-travail": "sciences-sociales",
    "alignement-ia": "ia",
    "biais-recompense": "ia",
    "metriques-de-vanite": "ia",
}

# domaine -> (bloc general injecte en repli, mots-cles d'ancrage lexical)
# Le repli n'apparaît QUE si (a) le score est dans la bande d'indice ET
# (b) la question contient un mot-clé du domaine du concept le plus
# proche : deux indices concordants, jamais un saut au hasard.
_BLOCS_DOMAINES: dict[str, tuple[str, tuple[str, ...]]] = {
    "philosophie": (
        "Cadre philosophique : problemes classiques (existence, liberte, "
        "connaissance, sens, verite), distinctions conceptuelles et "
        "references (Platon, Kant, Hume, Sartre). Repondre par une "
        "analyse nuancee, pas par un fait.",
        ("philosophie", "existentiel", "absurde", "liberte",
         "libre arbitre", "metaphysique", "sens de la vie",
         "connaissance", "verite")),
    "ethique": (
        "Cadre ethique : devoir contre consequence, justesse des regles, "
        "responsabilite des acteurs — exposer les principes en presence "
        "avant de trancher, sans dogmatisme.",
        ("ethique", "moral", "devoir", "responsab", "justice",
         "juste", "consequence", "liberte")),
    "logique": (
        "Cadre logique : premisses, validite contre veracite, lien de "
        "cause a effet, cas particuliers contre regle generale — "
        "raisonner pas a pas et signaler les limites du raisonnement.",
        ("logique", "preuve", "raison", "cause", "correlation",
         "syllogisme", "argument", "conclusion", "induction",
         "deduction", "coherent")),
    "psychologie": (
        "Cadre psychologique : mecanismes cognitifs et emotionnels "
        "(attention, motivation, biais, stress, habitudes) — decrire le "
        "fonctionnement de l'esprit, pas juger la personne.",
        ("psychologie", "stress", "emotion", "peur", "motivation",
         "habitude", "cognition", "cerveau", "comportement", "esprit",
         "anxiete", "procrastin")),
    "sciences-sociales": (
        "Cadre des sciences sociales : structures, inegalites, effets de "
        "systeme, usages du travail — expliquer les dynamiques "
        "collectives plutot que les choix individuels.",
        ("societe", "social", "inegalite", "economie", "travail",
         "culture", "institution", "collectif", "systeme")),
    "ia": (
        "Cadre de l'ia : objectifs, donnees, apprentissage, limites et "
        "risques des systemes — distinguer ce que le modele sait de ce "
        "qu'il improvise.",
        ("intelligence artificielle", "apprentissage", "algorithme",
         "donnees", "neurone", "modele", "robot", "generation")),
}

_charge_fait = False                # concepts.toml lu une seule fois


def repli_actif() -> bool:
    """AURA_CONCEPTS_REPLI != 0 (actif par defaut)."""
    return os.environ.get("AURA_CONCEPTS_REPLI", "1") != "0"


def _sans_accent(texte: str) -> str:
    brut = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in brut if not unicodedata.combining(c))


def _charger_toml() -> None:
    """Charge concepts.toml (racine du projet) dans _CONCEPTS/_DOMAINES.

    Idempotent (une seule lecture par process) ; fichier absent = no-op.
    Une cle TOML SURCHARGE le concept integre de meme nom (calibrage
    utilisateur), une nouvelle cle etend la bibliotheque.
    """
    global _charge_fait
    if _charge_fait:
        return
    _charge_fait = True
    libre = (os.environ.get("AURA_CONCEPTS_TOML") or "").strip()
    chemin = (Path(libre) if libre
              else Path(__file__).resolve().parent.parent / "concepts.toml")
    if not chemin.is_file():
        return
    try:
        with open(chemin, "rb") as f:
            donnees = tomllib.load(f)
    except Exception as e:  # noqa: BLE001 — fichier corrompu = defauts
        LOG.warning("[concepts] %s illisible (%s) -> bibliotheque integre",
                    chemin.name, e)
        return
    bloc = donnees.get("concepts")
    if not isinstance(bloc, dict):
        return
    ajouts = 0
    for nom, valeur in bloc.items():
        domaine = None
        if isinstance(valeur, str):
            description = valeur.strip()
        elif isinstance(valeur, dict):
            description = str(valeur.get("description") or "").strip()
            domaine = str(valeur.get("domaine") or "").strip()
        else:
            continue
        if not str(nom) or not description:
            continue
        _CONCEPTS[str(nom)] = description
        if domaine:
            _DOMAINES[str(nom)] = domaine
        ajouts += 1
    if ajouts:
        LOG.info("[concepts] %d entree(s) depuis %s -> %d au total",
                 ajouts, chemin.name, len(_CONCEPTS))


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
    _charger_toml()                 # concepts.toml = partie de la biblio
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

    Renvoie {"concept": nom, "score": float, "description": texte} ; en
    bande d'indice (>= _SEUIL_INDICE mais sous le seuil), un REPLI de
    domaine avec {"repli": True, "domaine": ...} si la question contient
    un mot-clé de ce domaine (zéro muet sinon).
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
    if score >= seuil():
        return {"concept": noms[i], "score": round(score, 3),
                "description": _CONCEPTS[noms[i]]}
    # REPLI DOMAIN (zéro muet) : sous le seuil mais deux indices
    # concordants (proximite vectorielle + mot-clé du domaine) -> un
    # rappel GENERAL du domaine remplace le silence. Sans les deux, rien
    # (les hors-sujet reels restent rejetes, cf. calibrage).
    if not repli_actif() or score < _SEUIL_INDICE:
        return None
    nom = noms[i]
    domaine = _DOMAINES.get(nom)
    if domaine is None:
        return None
    bloc, mots_cles = _BLOCS_DOMAINES.get(domaine, ("", ()))
    if not bloc or not any(m in _sans_accent(texte) for m in mots_cles):
        return None
    return {"concept": nom, "score": round(score, 3), "description": bloc,
            "repli": True, "domaine": domaine}


def ancrage(texte: str) -> str | None:
    """Bloc a injecter dans le prompt systeme, ou None.

    Court-circuit symbolique : le LLM recoit le lexique du concept des
    la premiere phrase, au lieu de partir de zero (ou d'inventer les
    termes).
    """
    hit = detecter(texte)
    if not hit:
        return None
    if hit.get("repli"):
        return (
            f"REPLI DOMAIN (approche : {hit['domaine']}, "
            f"similarite {hit['score']:.0%}) : {hit['description']}\n"
            "Repli large, pas un concept affirme : utilise ce cadre SEULEMENT "
            "si la question touche vraiment ce domaine, sinon reponds "
            "normalement sans forcer le hors-sujet.")
    return (
        f"LEXIQUE (concept rapproche : {hit['concept']}, "
        f"similarite {hit['score']:.0%}) : {hit['description']}\n"
        "Si la question touche ce concept, appuie-toi sur ce lexique "
        "(termes justes, nuances) sans forcer le hors-sujet.")
