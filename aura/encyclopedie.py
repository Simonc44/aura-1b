"""Expert ENCYCLOPEDIQUE : « qui est Steve Jobs ? », « c'est quoi un trou noir ? ».

Probleme
--------
Une question de culture generale partait au web (extraits DuckDuckGo
heterogenes) puis au 1B, qui reformule de memoire : fautes de date, noms
inventes. C'est la ou un 1B parait le plus petit.

Remede : le FAIT vient de Wikipedia (resume d'introduction), recopie tel
quel et sourcé — le modele ne reecrit rien, donc ne peut rien inventer.

Tolerance aux fautes
--------------------
« Steave Jobs », « Napoleon Bonnaparte », « trou noire » : la recherche
Wikipedia (CirrusSearch) corrige deja beaucoup. En plus, CHAQUE titre
candidat est compare au sujet demande (distance d'edition mot a mot) et on
n'accepte qu'une proximite >= 0,85. Quand le sujet a ete corrige, la reponse
le dit (« Je suppose que tu parles de **Steve Jobs**. ») au lieu de
corriger en silence.

Garde-fous (abstention plutot que faux — meme principe que `titulaires`)
-----------------------------------------------------------------------
- forme de question etroite : « qui est/etait X », « c'est qui X »,
  « qu'est-ce que X », « c'est quoi X », « parle-moi de X », « presente-moi X » ;
- tout ce qui est relatif a l'ACTUALITE, a un superlatif ou a une relation
  (« le meilleur… », « le fondateur de… », « le president de… »,
  « actuellement ») passe son chemin : ce n'est pas une notice ;
- pronoms (« qui est tu », « c'est quoi ca ») : jamais ;
- page d'homonymie, extrait trop court, titre trop eloigne : abstention ;
- reseau coupe, `AURA_APIS=0` ou `AURA_ENCYCLO=0` : None, la cascade continue.

Stdlib seule ; passe par `apis_publiques._json` (cache TTL, timeout court,
echec memorise, jamais d'exception).
"""
from __future__ import annotations

import difflib
import logging
import os
import re
import time
import unicodedata
import urllib.parse

from . import apis_publiques

LOG = logging.getLogger("aura.encyclopedie")

_API = "https://fr.wikipedia.org/w/api.php"
_WIKI = "https://fr.wikipedia.org/wiki/"
_BUDGET_S = 6.0            # plafond de temps cumule (appels enchaines)
_SEUIL = 0.85              # proximite minimale sujet <-> titre
_MAX_CARACTERES = 700

# ── forme de la question ───────────────────────────────────────────────────
_POLI = re.compile(
    r"^(?:(?:dis|dites)[-\s]moi|peux[-\s]tu\s+me\s+dire|"
    r"pouvez[-\s]vous\s+me\s+dire|tu\s+peux\s+me\s+dire|"
    r"je\s+voudrais\s+savoir|j['’]aimerais\s+savoir|et|mais|alors|"
    r"dis|svp|stp)[,\s]+", re.IGNORECASE)

_FORMES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"qui\s+(?:est|[ée]tait|fut|a\s+[ée]t[ée])\s+(?P<s>[^?!.;,]+)",
    r"c['’]est\s+qui\s+(?P<s>[^?!.;,]+)",
    r"who\s+(?:is|was)\s+(?P<s>[^?!.;,]+)",
    r"qu['’]est[-\s]ce\s+(?:que|qu['’])\s*(?:c['’]est\s+)?(?P<s>[^?!.;,]+)",
    r"c['’]est\s+quoi\s+(?P<s>[^?!.;,]+)",
    r"(?:parle|parlez)[-\s]moi\s+(?:de|d['’]|du|des)\s*(?P<s>[^?!.;,]+)",
    r"(?:pr[ée]sente|pr[ée]sentez)[-\s]moi\s+(?P<s>[^?!.;,]+)",
))

_QUEUE = re.compile(
    r"\s*(?:s['’]il\s+(?:te|vous)\s+pla[iî]t|stp|svp|"
    r"en\s+(?:quelques\s+mots|bref|r[ée]sum[ée])|exactement|vraiment)\s*$",
    re.IGNORECASE)
_ARTICLE = re.compile(
    r"^(?:l['’]|le\s+|la\s+|les\s+|un\s+|une\s+|des\s+|du\s+|de\s+la\s+|d['’])\s*",
    re.IGNORECASE)

_STOP = frozenset({"le", "la", "les", "l", "de", "du", "des", "d", "un", "une",
                   "et", "en", "au", "aux"})
# Mots (sans accents) qui sortent la question du cadre « notice encyclopedique »
_PRONOMS = frozenset(
    "tu vous toi moi il elle ils elles ce cela ca on nous je quoi qui que "
    "ton ta tes votre vos mon ma mes son sa ses leur leurs "
    # nom de l'assistant : « qui est aura » n'est pas une notice
    "aura rontogpt".split())
_RELATIONNEL = frozenset(
    "president presidente premier premiere ministre roi reine maire chef "
    "directeur patron pdg champion capitale population prix meteo createur "
    "auteur fondateur inventeur developpeur".split())
_ACTUALITE = frozenset(
    "meilleur meilleure meilleurs meilleures pire plus moins dernier derniere "
    "actuel actuelle actuellement aujourd hui maintenant recent recente "
    "nouveau nouvelle gagnant vainqueur".split())
_ANNEE_RECENTE = re.compile(r"\b20(?:2[4-9]|[3-9]\d)\b")
_CHIFFRE = re.compile(r"\d")    # « Napoleon 1er », « 42 » : on s'abstient
_COMPOSE = re.compile(r"\b(?:et|ou|vs|versus|contre)\b", re.IGNORECASE)


def _sans_accents(texte: str) -> str:
    n = unicodedata.normalize("NFKD", (texte or "").lower())
    return "".join(c for c in n if not unicodedata.combining(c))


def _mots(texte: str) -> list[str]:
    """Mots normalises (sans accents, sans articles, sans parentheses)."""
    texte = re.sub(r"\([^)]*\)", " ", texte or "")
    return [m for m in re.findall(r"[a-z0-9]+", _sans_accents(texte))
            if m not in _STOP]


def _actif() -> bool:
    return (os.environ.get("AURA_ENCYCLO", "1") != "0"
            and apis_publiques._actives())


def analyser(question: str) -> dict | None:
    """La question est-elle une demande de notice ? Sinon None. Pur (0 reseau)."""
    q = " ".join((question or "").split())
    if not q or len(q) > 120:
        return None
    q = _POLI.sub("", q)
    sujet = ""
    for forme in _FORMES:
        m = forme.match(q)
        if m:
            sujet = m.group("s")
            break
    if not sujet:
        return None
    sujet = _QUEUE.sub("", sujet).strip(" «»\"'")
    recherche = _ARTICLE.sub("", sujet).strip()
    mots = _mots(recherche)
    if not mots or len(recherche.split()) > 6 or len(recherche) > 60:
        return None
    if mots[0] in _PRONOMS or mots[0] in _RELATIONNEL:
        return None
    if set(mots) & _ACTUALITE or _ANNEE_RECENTE.search(recherche):
        return None
    if _CHIFFRE.search(recherche):
        return None             # « Napoleon 1er », « 42 » : on s'abstient
    if _COMPOSE.search(recherche):
        return None             # comparaison / question composee : on passe
    return {"sujet": sujet, "sujet_de": _sans_accents(sujet).lower(), "recherche": recherche}


def sujet_de(question: str) -> str | None:
    """Sujet normalise (sans accents, minuscules) si la question est une notice.

    Renvoie None si la question ne correspond pas a une demande encyclopedique.
    Utile pour le pipeline : `encyclopedie.sujet_de(question) is not None` teste
    si la question merite une consultation Wikipedia.
    """
    cible = analyser(question)
    return _sans_accents(cible["recherche"]).lower() if cible else None


# ── proximite sujet <-> titre ──────────────────────────────────────────────
def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _present(mot: str, titre: list[str]) -> bool:
    """Le mot (a une faute pres) figure-t-il dans le titre ?"""
    if mot in titre:
        return True
    if len(mot) <= 3:
        return False            # mots courts : egalite stricte
    return any(len(t) > 3 and _ratio(mot, t) >= 0.8 for t in titre)


def proximite(sujet: str, titre: str) -> float:
    """0..1 : a quel point `titre` est le sujet demande (fautes tolerees)."""
    a, b = _mots(sujet), _mots(titre)
    if not a or not b:
        return 0.0
    note = _ratio(" ".join(a), " ".join(b))
    if all(_present(m, b) for m in a):
        # tous les mots du sujet sont dans le titre : « Einstein » ->
        # « Albert Einstein ». Un titre plus long vaut un peu moins qu'un
        # titre identique, pour que « Paris » batte « Paris Hilton ».
        note = max(note, 1.0 if len(a) >= len(b) else 0.88)
    return note


def _corrige(sujet: str, titre: str) -> bool:
    """Le sujet a-t-il ete CORRIGE (faute) et non simplement complete ?"""
    a, b = _mots(sujet), _mots(titre)
    return not all(m in b for m in a)


def _proche(sujet: str, titre: str) -> bool:
    """Vrai si `titre` est une proximite acceptable de `sujet` (seuil _SEUIL)."""
    return proximite(sujet, titre) >= _SEUIL


# ── Wikipedia ──────────────────────────────────────────────────────────────

def _url(**params) -> str:
    return _API + "?" + urllib.parse.urlencode(
        {**params, "format": "json", "formatversion": 2})


def _chercher(terme: str) -> tuple[list[str], str]:
    """(titres candidats dans l'ordre de pertinence, suggestion orthographique)."""
    donnee = apis_publiques._json(_url(
        action="query", list="search", srsearch=terme, srlimit=5,
        srnamespace=0, srprop="", srinfo="suggestion"))
    requete = donnee.get("query") if isinstance(donnee, dict) else None
    if not isinstance(requete, dict):
        return [], ""
    titres = [str(r["title"]) for r in (requete.get("search") or [])
              if isinstance(r, dict) and r.get("title")]
    info = requete.get("searchinfo")
    suggestion = str(info.get("suggestion") or "") if isinstance(info, dict) else ""
    return titres, suggestion


def _page(titre: str) -> dict | None:
    donnee = apis_publiques._json(_url(
        action="query", prop="extracts|pageprops", exintro=1, explaintext=1,
        exsentences=3, redirects=1, ppprop="disambiguation", titles=titre))
    requete = donnee.get("query") if isinstance(donnee, dict) else None
    pages = requete.get("pages") if isinstance(requete, dict) else None
    if not isinstance(pages, list) or not pages:
        return None
    page = pages[0]
    return page if isinstance(page, dict) and not page.get("missing") else None


def _nettoyer(extrait: str) -> str:
    t = " ".join((extrait or "").split())
    # prononciation / ecoute : du bruit pour un resume
    t = re.sub(r"\s*\((?:[^()]*(?:prononc|écouter|ecouter|ⓘ)[^()]*)\)", "", t,
               flags=re.IGNORECASE)
    if len(t) > _MAX_CARACTERES:
        coupe = t.rfind(". ", 0, _MAX_CARACTERES)
        t = t[:coupe + 1] if coupe > 150 else t[:_MAX_CARACTERES].rstrip() + "…"
    return t


def _lien(titre: str) -> str:
    return _WIKI + urllib.parse.quote(titre.replace(" ", "_"), safe="_()',")# ── point d'entree ─────────────────────────────────────────────────────────
def repondre(question: str) -> str | None:
    """Resume Wikipedia sourcé, ou None (abstention)."""
    if not _actif():
        return None
    cible = analyser(question)
    if cible is None:
        return None
    debut = time.time()

    def _reste() -> bool:
        return time.time() - debut < _BUDGET_S

    sujet = cible["recherche"]
    titres, suggestion = _chercher(sujet)
    classes = sorted(((proximite(sujet, t), -i, t) for i, t in enumerate(titres)),
                     reverse=True)
    if (not classes or classes[0][0] < _SEUIL) and suggestion and _reste():
        # la recherche n'a rien de proche : on retente avec la correction
        # orthographique proposee par Wikipedia (« steave jobs » -> « steve jobs »)
        titres2, _ = _chercher(suggestion)
        classes = sorted(
            ((proximite(sujet, t), -i, t) for i, t in enumerate(titres2)),
            reverse=True) or classes
    retenus = [c for c in classes if c[0] >= _SEUIL][:2]
    for _, _, titre in retenus:
        if not _reste():
            break
        page = _page(titre)
        if page is None:
            continue
        if (page.get("pageprops") or {}).get("disambiguation") is not None:
            continue            # « Martin » : plusieurs sens, on ne tranche pas
        extrait = _nettoyer(str(page.get("extract") or ""))
        if len(extrait) < 40 or "peut désigner" in extrait[:200].lower():
            continue
        titre_final = str(page.get("title") or titre)
        note = ""
        if _corrige(sujet, titre_final):
            note = f"Je suppose que tu parles de **{titre_final}**.\n\n"
        LOG.info("[encyclopedie] %r -> %s", sujet, titre_final)
        return (f"{note}{extrait}\n\n"
                f"Source : [Wikipédia — {titre_final}]({_lien(titre_final)}).")
    LOG.info("[encyclopedie] abstention pour %r", sujet)
    return None


def est_reponse_encyclopedie(texte: str) -> bool:
    """Vrai si `texte` sort de `resoudre` (lien Wikipedia en fin de reponse)."""
    return f"]({_WIKI}" in (texte or "")
