"""Titulaire ACTUEL d'une fonction : « qui est le président de la France ? ».

Probleme mesure (11 oct. 2026)
------------------------------
« qui est le président de la France » partait au web, DuckDuckGo rendait des
extraits generiques (« Le président de la République française est le chef de
l'État... ») qui ne contiennent pas le NOM, et le 1B recopiait l'extrait : une
reponse qui ne repond pas a la question.

Remede : un expert dedie, qui lit le titulaire dans Wikidata (donnees
structurees, sans cle, meme principe que `apis_publiques`) :

1. la FONCTION (« président de la France ») est cherchee comme entite ; si un
   alias/libelle correspond EXACTEMENT, on lit la propriete P1308
   (« titulaire ») ;
2. sinon, pour un pays ou une ville, on lit P35 (chef d'Etat) ou P6 (chef du
   gouvernement) sur le lieu lui-meme ;
3. parmi les mandats, on ne garde que celui qui est EN COURS (rang prefere,
   pas de date de fin passee), le plus recent en cas de doute.

Garde-fous (abstention plutot que faux) :

- la question doit avoir exactement la forme « qui est <article> <fonction>
  <prep> <lieu> » avec une fonction connue : tout le reste passe son chemin ;
- une annee dans la question (« en 2012 ») = question historique : on s'abstient ;
- une correspondance APPROXIMATIVE de nom n'est jamais retenue (« président du
  Sénat » ne repond pas pour « président de la France ») ;
- deux fonctions candidates qui donnent deux personnes differentes : abstention ;
- reseau coupe, `AURA_APIS=0`, entite introuvable : None, la cascade continue.

Stdlib seule ; passe par `apis_publiques._json` (cache TTL, timeout court,
echec memorise 2 min, jamais d'exception).
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
import urllib.parse

from . import apis_publiques

LOG = logging.getLogger("aura.titulaires")

_API = "https://www.wikidata.org/w/api.php"
_WIKI = "https://www.wikidata.org/wiki/"
_QID = re.compile(r"^Q\d+$")
_BUDGET_S = 8.0          # plafond de temps cumule (appels reseau enchaines)

_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre")

# ── la question ────────────────────────────────────────────────────────────
_ADV = (r"(?:(?:actuellement|en ce moment|aujourd['’]hui|maintenant|"
        r"[àa] l['’]heure actuelle)\s+)?")
_ROLE = (r"pr[ée]sidente?|premi[èe]re?\s+ministre|chancelier|chanceli[èe]re|"
         r"roi|reine|empereur|imp[ée]ratrice|maire|"
         r"chef\s+de\s+l['’]\s?[ée]tat|chef\s+du\s+gouvernement")
_PREP = r"de\s+la|de\s+l['’]|du|des|d['’]|de"

_RE_QUESTION = re.compile(
    r"\b(?:qui\s+est|c['’]est\s+qui)\s+" + _ADV
    + r"(?P<art>le|la|l['’])\s*(?P<role>" + _ROLE + r")\s+"
    r"(?P<prep>" + _PREP + r")\s*(?P<sujet>[^?!.;,]+)",
    re.IGNORECASE)

_ADV_FIN = re.compile(
    r"\s+(?:actuellement|aujourd['’]hui|maintenant|en ce moment|"
    r"[àa] l['’]heure actuelle)\s*$", re.IGNORECASE)
_ANNEE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")

# fonction -> propriete portee par le PAYS / la VILLE
_PROPRIETES = (
    (re.compile(r"pr[ée]sident|^roi$|^reine$|empereur|imp[ée]ratrice|"
                r"chef\s+de\s+l['’]\s?[ée]tat", re.IGNORECASE), "P35"),
    (re.compile(r"premi[èe]re?\s+ministre|chancel|^maire$|"
                r"chef\s+du\s+gouvernement", re.IGNORECASE), "P6"),
)
# description Wikidata d'un lieu (pays, Etat, commune...) : evite de prendre
# un film ou un prenom qui porterait le meme nom que le pays
_DESC_LIEU = re.compile(
    r"pays|[ée]tat|commune|ville|r[ée]publique|royaume|principaut[ée]|"
    r"territoire|nation|capitale|country|state|city|municipality|capital|"
    r"kingdom|republic|sovereign", re.IGNORECASE)

_MOTS_VIDES = frozenset({"le", "la", "les", "l", "de", "du", "des", "d",
                         "un", "une"})


def _cle(texte: str) -> str:
    """Forme de comparaison : sans accents, minuscule, sans articles."""
    n = unicodedata.normalize("NFKD", (texte or "").lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    return " ".join(m for m in re.findall(r"[a-z0-9]+", n)
                    if m not in _MOTS_VIDES)


def _propriete(role: str) -> str | None:
    for motif, prop in _PROPRIETES:
        if motif.search(role.strip()):
            return prop
    return None


def analyser(question: str) -> dict | None:
    """La question est-elle « qui est <fonction> <prep> <lieu> » ? Sinon None.

    Pur (aucun reseau) : sert aussi de detecteur a bas cout.
    """
    m = _RE_QUESTION.search(question or "")
    if m is None:
        return None
    sujet = _ADV_FIN.sub("", m.group("sujet")).strip(" «»\"'")
    if len(sujet) < 2 or len(sujet.split()) > 5:
        return None
    if _ANNEE.search(sujet) or re.search(r"\b(?:et|ou)\b", sujet,
                                         re.IGNORECASE):
        return None             # question historique ou composee : on passe
    prep = m.group("prep")
    colle = prep.endswith(("'", "’"))
    syntagme = (question[m.start("art"):m.start("prep")]
                + prep + ("" if colle else " ") + sujet)
    syntagme = " ".join(syntagme.split())
    requete = " ".join((m.group("role") + " " + prep).split()) \
        + ("" if colle else " ") + sujet
    return {"syntagme": syntagme, "role": m.group("role"),
            "requete": requete, "sujet": sujet}


# ── Wikidata ───────────────────────────────────────────────────────────────

def _url(**params) -> str:
    return _API + "?" + urllib.parse.urlencode({**params, "format": "json"})


def _chercher(terme: str) -> list[dict]:
    donnee = apis_publiques._json(_url(
        action="wbsearchentities", search=terme, language="fr",
        uselang="fr", type="item", limit=7))
    trouves = donnee.get("search") if isinstance(donnee, dict) else None
    return [r for r in (trouves or [])
            if isinstance(r, dict) and _QID.match(str(r.get("id", "")))]


def _texte_correspondant(resultat: dict) -> str:
    match = resultat.get("match")
    if isinstance(match, dict) and match.get("text"):
        return str(match["text"])
    return str(resultat.get("label") or "")


def _mandats(qid: str, prop: str) -> list[dict]:
    """Les mandats d'UNE propriete (wbgetclaims : on ne telecharge pas les
    milliers de declarations d'un pays)."""
    donnee = apis_publiques._json(_url(
        action="wbgetclaims", entity=qid, property=prop))
    claims = donnee.get("claims") if isinstance(donnee, dict) else None
    liste = (claims or {}).get(prop) if isinstance(claims, dict) else None
    return [s for s in (liste or []) if isinstance(s, dict)]


def _libelle(qid: str) -> str:
    donnee = apis_publiques._json(_url(
        action="wbgetentities", ids=qid, props="labels",
        languages="fr|en", languagefallback=1))
    entites = donnee.get("entities") if isinstance(donnee, dict) else None
    entite = (entites or {}).get(qid) if isinstance(entites, dict) else None
    labels = entite.get("labels") if isinstance(entite, dict) else None
    if not isinstance(labels, dict):
        return ""
    for langue in ("fr", "en"):
        v = labels.get(langue)
        if isinstance(v, dict) and v.get("value"):
            return str(v["value"])
    for v in labels.values():
        if isinstance(v, dict) and v.get("value"):
            return str(v["value"])
    return ""


# ── choix du mandat EN COURS ───────────────────────────────────────────────

def _qid(statement: dict) -> str:
    snak = statement.get("mainsnak") or {}
    if snak.get("snaktype") != "value":
        return ""
    valeur = (snak.get("datavalue") or {}).get("value")
    ident = valeur.get("id") if isinstance(valeur, dict) else ""
    return str(ident) if _QID.match(str(ident or "")) else ""


def _quand(statement: dict, prop: str) -> tuple[int, int, int, int] | None:
    """(annee, mois, jour, precision) d'un qualificatif de date, sinon None."""
    for q in (statement.get("qualifiers") or {}).get(prop, []):
        valeur = ((q or {}).get("datavalue") or {}).get("value") or {}
        m = re.match(r"([+-])(\d+)-(\d\d)-(\d\d)", str(valeur.get("time", "")))
        if not m:
            continue
        annee = int(m.group(2)) * (1 if m.group(1) == "+" else -1)
        return (annee, int(m.group(3)), int(m.group(4)),
                int(valeur.get("precision") or 11))
    return None


def _aujourdhui() -> tuple[int, int, int]:
    t = time.gmtime()
    return t.tm_year, t.tm_mon, t.tm_mday


def _termine(statement: dict) -> bool:
    if "P582" not in (statement.get("qualifiers") or {}):
        return False
    fin = _quand(statement, "P582")
    if fin is None:
        return True              # fin « inconnue » : le mandat a bien pris fin
    return (fin[0], fin[1] or 1, fin[2] or 1) <= _aujourdhui()


def _courant(mandats: list[dict]) -> dict | None:
    """Le mandat en cours : prefere d'abord, sinon sans fin passee ; le plus
    recent en cas de doute. None si tous les mandats sont termines."""
    valides = [s for s in mandats
               if s.get("rank") != "deprecated" and _qid(s)
               and not _termine(s)]
    prefere = [s for s in valides if s.get("rank") == "preferred"]
    candidats = prefere or valides
    if not candidats:
        return None
    return max(candidats, key=lambda s: (_quand(s, "P580") or (0, 0, 0, 0))[:3])


def _date_fr(quand: tuple[int, int, int, int]) -> str:
    annee, mois, jour, precision = quand
    if precision >= 11 and mois and jour:
        return f"le {'1er' if jour == 1 else jour} {_MOIS[mois - 1]} {annee}"
    if precision >= 10 and mois:
        return f"{_MOIS[mois - 1]} {annee}"
    return str(annee)


# ── point d'entree ─────────────────────────────────────────────────────────

def resoudre(question: str) -> str | None:
    """« Le président de la France est Emmanuel Macron (en fonction depuis le
    14 mai 2017). Source : [Wikidata](...). » — ou None (abstention)."""
    if not apis_publiques._actives():
        return None
    cible = analyser(question)
    if cible is None:
        return None
    debut = time.time()

    def _temps_restant() -> bool:
        return time.time() - debut < _BUDGET_S

    trouve: tuple[str, str, tuple[int, int, int, int] | None] | None = None

    # 1. la FONCTION elle-meme : correspondance EXACTE de nom, jamais floue
    cle_requete = _cle(cible["requete"])
    titulaires: dict[str, tuple[str, tuple | None]] = {}
    for r in _chercher(cible["requete"]):
        if _cle(_texte_correspondant(r)) != cle_requete:
            continue
        if not _temps_restant() or len(titulaires) > 1:
            break
        mandat = _courant(_mandats(str(r["id"]), "P1308"))
        if mandat is not None:
            titulaires.setdefault(_qid(mandat),
                                  (str(r["id"]), _quand(mandat, "P580")))
    if len(titulaires) == 1:
        (qid_t, (qid_src, depuis)), = titulaires.items()
        trouve = (qid_t, qid_src, depuis)

    # 2. a defaut, le LIEU : chef d'Etat (P35) ou de gouvernement (P6)
    if trouve is None and _temps_restant():
        prop = _propriete(cible["role"])
        if prop is not None:
            cle_sujet = _cle(cible["sujet"])
            lieux = [r for r in _chercher(cible["sujet"])
                     if _cle(_texte_correspondant(r)) == cle_sujet
                     and _DESC_LIEU.search(str(r.get("description") or ""))]
            if lieux and _temps_restant():
                mandat = _courant(_mandats(str(lieux[0]["id"]), prop))
                if mandat is not None:
                    trouve = (_qid(mandat), str(lieux[0]["id"]),
                              _quand(mandat, "P580"))

    if trouve is None or not _temps_restant():
        LOG.info("[titulaires] abstention pour %r", cible["syntagme"])
        return None
    qid_t, qid_src, depuis = trouve
    nom = _libelle(qid_t)
    if not nom:
        return None
    syntagme = cible["syntagme"]
    phrase = syntagme[:1].upper() + syntagme[1:]
    since = f" (en fonction depuis {_date_fr(depuis)})" if depuis else ""
    LOG.info("[titulaires] %s -> %s", cible["syntagme"], nom)
    return f"{phrase} est {nom}{since}. Source : [Wikidata]({_WIKI}{qid_src})."


def est_reponse_titulaire(texte: str) -> bool:
    """Vrai si `texte` sort de `resoudre` (lien Wikidata en fin de reponse)."""
    return f"]({_WIKI}" in (texte or "")
