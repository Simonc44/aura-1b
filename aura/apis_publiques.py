"""Connecteurs d'API publiques — liste de référence public-apis/public-apis.

Cinq API retenues dans la liste officielle github.com/public-apis/public-apis,
toutes **sans clé** et **sans dépendance** (stdlib `urllib` seul), toutes
mesurées < 1 s au premier appel :

- Open-Meteo  (rubrique Weather)          → météo actuelle d'une ville
- Frankfurter (Currency Exchange)         → taux de change ECB, conversion
- Wiktionary   (Books & Writing)           → définitions françaises
- Open Library (Books)                     → livres, auteurs, années
- Nominatim    (Geocoding, OpenStreetMap)  → localisation, coordonnées

REST Countries était prévu pour les capitales/populations : l'API exige
désormais une clé (401 sur api.restcountries.com) — écartée pour rester
sans secret. Les autres entrées de la liste exigent une clé ou sont lentes.

Principe « hyper léger et rapide » :

- `AURA_APIS=0` coupe la brique (défaut : actif) ;
- la DÉTECTION est une regex locale : zéro octet envoyé quand la question
  ne concerne aucune API (le banc, les puzzles, le calcul verbal ne voient
  rien changer) ;
- timeout court `AURA_API_TIMEOUT` (2,5 s par appel), corps borné 300 Ko ;
- cache mémoire `AURA_API_CACHE_S` (600 s) par URL + échec mémorisé 120 s
  (hors ligne : une tentative toutes les 2 minutes, pas une par question) ;
- jamais bloquant : erreur = texte vide, la cascade existante répond comme
  avant ;
- exécuté EN PARALLÈLE du web/PGS par l'orchestrateur (`_executer_experts`)
  : la latence réseau se cache sous celle de DuckDuckGo.
"""
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, NamedTuple

LOG = logging.getLogger("aura.apis")

# User-Agent identifiable (politique d'usage Nominatim / Wikimedia)
_UA = ("Mozilla/5.0 (compatible; aura-1b/1; "
       "https://github.com/Simonc44/aura-1b)")

_DELAI_ECHEC_S = 120.0       # un echec rebloque l'URL 2 min
_MAX_CACHE = 512             # purge grossiere anti-fuite

_cache: dict[str, tuple[float, Any]] = {}      # url -> (horodatage, donnees)
_echecs: dict[str, float] = {}                 # url -> horodatage echec
_verrou = threading.Lock()


# ── socle : activation, timeouts, GET JSON cache ─────────────────────────

def _actives() -> bool:
    """Vrai si la brique n'est pas coupee (AURA_APIS=0)."""
    return os.environ.get("AURA_APIS", "1") != "0"


def _timeout() -> float:
    """Timeout par appel (AURA_API_TIMEOUT, defaut 2,5 s)."""
    try:
        return max(0.2, float(os.environ.get("AURA_API_TIMEOUT", "2.5")))
    except ValueError:
        return 2.5


def _duree_cache() -> float:
    """TTL du cache (AURA_API_CACHE_S, defaut 600 s)."""
    try:
        return max(0.0, float(os.environ.get("AURA_API_CACHE_S", "600")))
    except ValueError:
        return 600.0


def _json(url: str) -> Any | None:
    """GET JSON borne : cache TTL, echec memorise, jamais d'exception.

    Reponse deja en cache = 0 octet. URL en echec = 0 octet pendant 120 s
    (une question hors ligne ne paie le timeout qu'une fois).
    """
    if not _actives():
        return None
    maintenant = time.time()
    with _verrou:
        entree = _cache.get(url)
        if entree is not None and maintenant - entree[0] < _duree_cache():
            return entree[1]
        echec = _echecs.get(url)
        if echec is not None and maintenant - echec < _DELAI_ECHEC_S:
            return None
    try:
        requete = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(requete, timeout=_timeout()) as reponse:
            corps = reponse.read(300_000).decode("utf-8", "replace")
        donnees = json.loads(corps)
    except Exception as e:                    # reseau off, 404, JSON invalide
        LOG.info("[api] %s : %s", url.split("?")[0], e)
        with _verrou:
            if len(_echecs) > _MAX_CACHE:
                _echecs.clear()
            _echecs[url] = time.time()
        return None
    with _verrou:
        if len(_cache) > _MAX_CACHE:          # purge des entrees expirees
            seuil = time.time() - _duree_cache()
            for clee in [k for k, (t, _) in _cache.items() if t < seuil]:
                del _cache[clee]
        _cache[url] = (time.time(), donnees)
    return donnees


def _g(valeur: Any) -> str:
    """Nombre au format FR court : 89.09 -> « 89,09 »."""
    try:
        return format(float(valeur), ".6g").replace(".", ",")
    except (TypeError, ValueError):
        return "?"


def _decimal(valeur: Any, decimales: int) -> str:
    """Nombre a N decimales, virgule francaise : 15.34 -> « 15,3 »."""
    try:
        return f"{float(valeur):.{decimales}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "?"


# ── 1. Open-Meteo : geocodage + meteo actuelle ───────────────────────────

_MOTIF_METEO = re.compile(
    r"m[ée]t[ée]o\b|quel(?:le)? temps|temp[ée]rature",
    re.IGNORECASE)

# mots qui coupent la ville (« meteo de Lyon demain » -> « Lyon »)
_INTERRUPTIONS = frozenset({
    "demain", "aujourd'hui", "aujourd’hui", "hier", "maintenant", "ici",
    "matin", "soir", "nuit", "apres", "après", "semaine", "week-end",
    "?", "!", ".",
})

_CIEL = {
    0: "ciel dégagé", 1: "peu nuageux", 2: "partiellement nuageux",
    3: "ciel couvert", 45: "brouillard", 48: "brouillard givrant",
    51: "bruine légère", 53: "bruine", 55: "bruine dense",
    56: "bruine givrante", 57: "bruine givrante dense",
    61: "pluie légère", 63: "pluie", 65: "pluie forte",
    66: "pluie verglaçante", 67: "pluie verglaçante forte",
    71: "neige légère", 73: "neige", 75: "neige forte",
    77: "grains de neige",
    80: "averses légères", 81: "averses", 82: "averses fortes",
    85: "averses de neige", 86: "averses de neige fortes",
    95: "orage", 96: "orage à la grêle", 99: "orage violent à la grêle",
}


def _ville(suite: str) -> str:
    """« fait-il a Paris ? » -> « Paris » ; « de Lyon demain » -> « Lyon »."""
    m = re.search(r"\b(?:de|du|des|à|a|pour|sur)\s+(.+)", suite, re.IGNORECASE)
    reste = m.group(1) if m else suite
    retenus: list[str] = []
    for brut in re.split(r"[\s?!,;:]+", reste.strip())[:6]:
        mot = brut.strip("«»'\"()[]{}")
        if not mot or mot.lower() in _INTERRUPTIONS:
            break
        retenus.append(mot)
        if len(retenus) >= 3:
            break
    return " ".join(retenus)


def _meteo(question: str) -> str:
    m = _MOTIF_METEO.search(question)
    if not m:
        return ""
    ville = _ville(question[m.end():])
    if len(re.sub(r"[^A-Za-zÀ-ÿ]", "", ville)) < 3:
        return ""                       # pas de ville exploitabile
    geo = _json("https://geocoding-api.open-meteo.com/v1/search?"
                + urllib.parse.urlencode({
                    "name": ville, "count": 1,
                    "language": "fr", "format": "json"}))
    if not geo or not geo.get("results"):
        return ""
    lieu = geo["results"][0]
    previsions = _json(
        "https://api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode({
            "latitude": lieu.get("latitude", 0),
            "longitude": lieu.get("longitude", 0),
            "current": ("temperature_2m,apparent_temperature,weather_code,"
                        "wind_speed_10m"),
            "timezone": "auto"}))
    if not previsions or not isinstance(previsions, dict) \
            or "current" not in previsions:
        return ""
    c = previsions["current"]
    endroit = str(lieu.get("name") or ville)
    pays = lieu.get("country") or lieu.get("country_code")
    if pays:
        endroit += f", {pays}"
    heure = str(c.get("time") or "")[11:16]
    ciel = _CIEL.get(c.get("weather_code"), "temps variable")
    return (f"Météo à {endroit}"
            + (f" ({heure})" if heure else "")
            + f" : {_decimal(c.get('temperature_2m'), 1)} °C, ressenti "
              f"{_decimal(c.get('apparent_temperature'), 1)} °C, {ciel}, "
              f"vent {_decimal(c.get('wind_speed_10m'), 0)} km/h.")


# ── 2. Frankfurter : taux de change ECB ──────────────────────────────────

_ISO = frozenset(
    "AUD BGN BRL CAD CHF CNY CZK DKK EUR GBP HKD HUF IDR ILS INR ISK JPY "
    "KRW MXN MYR NOK NZD PHP PLN RON SEK SGD THB TRY USD ZAR".split())

# lexique francais -> ISO (« livre » seul absent : collision avec le livre)
_LEXIQUE: dict[str, str] = {
    "€": "EUR", "euro": "EUR", "euros": "EUR",
    "$": "USD", "dollar": "USD", "dollars": "USD",
    "livre sterling": "GBP", "livres sterling": "GBP",
    "franc suisse": "CHF", "francs suisses": "CHF", "franc suisse": "CHF",
    "yen": "JPY", "yens": "JPY", "yuan": "CNY", "yuans": "CNY",
    "roupie": "INR", "roupies": "INR", "real": "BRL", "reals": "BRL",
    "peso": "MXN", "pesos": "MXN", "couronne": "SEK", "couronnes": "SEK",
    "zloty": "PLN", "zlotys": "PLN", "won": "KRW", "wons": "KRW",
}

_LEXIQUE_RE = (
    r"euros?|dollars?|livres? sterling|francs? suiss[ea]s?|yens?|yuans?|"
    r"roupies?|reals?|pesos?|couronnes?|zlotys?|wons?|€|\$|£|"
    + "|".join(sorted(_ISO, key=len, reverse=True)))

_MONTANT = r"\d+(?:[.,]\d+)?"
# deux pieces DEVISES cote a cote : « 100 dollars en euros » — une seule
# devise (« je dépense 4 euros ») ne peut PAS declencher le connecteur
_CONV1 = re.compile(
    rf"({_MONTANT})\s*({_LEXIQUE_RE})\s*(?:en|vers|→|pour|=)\s*({_LEXIQUE_RE})",
    re.IGNORECASE)
_CONV2 = re.compile(
    rf"taux de change\s+({_LEXIQUE_RE})\s*(?:en|vs|vers|contre|/)\s*({_LEXIQUE_RE})",
    re.IGNORECASE)


def _devise_iso(jeton: str) -> str | None:
    """« $ » -> USD, « euros » -> EUR, « usd » -> USD (ISO deja connu)."""
    iso = _LEXIQUE.get(jeton.strip().lower())
    if iso:
        return iso
    code = jeton.strip().upper()
    if code in _ISO:
        return code
    return None


def _devises(question: str) -> str:
    m1 = _CONV1.search(question)
    if m1 is not None:
        brut, de, vers = m1.group(1), m1.group(2), m1.group(3)
        montant = float(brut.replace(",", "."))
    else:
        m2 = _CONV2.search(question)
        if m2 is None:
            return ""
        de, vers = m2.group(1), m2.group(2)
        montant = 1.0
    iso_de, iso_vers = _devise_iso(de), _devise_iso(vers)
    if not iso_de or not iso_vers or iso_de == iso_vers:
        return ""
    donnee = _json("https://api.frankfurter.app/latest?"
                   + urllib.parse.urlencode({
                       "amount": format(montant, ".6g"),
                       "from": iso_de, "to": iso_vers}))
    if not donnee or not isinstance(donnee, dict):
        return ""
    valeur = (donnee.get("rates") or {}).get(iso_vers)
    if valeur is None:
        return ""
    return (f"{_g(montant)} {iso_de} = {_g(valeur)} {iso_vers} "
            f"(taux de change ECB du {donnee.get('date', '?')}).")


# ── 3. Wiktionary : definitions francaises ────────────────────────────────

_MOTIF_DEFI = re.compile(
    r"d[ée]finition\s+(?:de|du|des|d'|d’)"
    r"|signification\s+(?:de|du|des|d'|d’)"
    r"|que signifie|que veut dire|quel(?:le)? est le sens de",
    re.IGNORECASE)

_TETE = frozenset({
    "le", "la", "les", "un", "une", "de", "du", "des", "ce", "cet",
    "cette", "mon", "ton", "son", "mot", "est", "quel", "quelle",
})


def _mot_recherche(suite: str, max_mots: int = 5,
                   tete: bool = True) -> str:
    """Sujet de la recherche : « la photosynthese ? » -> « photosynthese ».

    tete=False garde les articles (les titres de livres en ont besoin :
    « Le Petit Prince »).
    """
    phrase = re.split(r"[?!.;:]", suite.strip())[0]
    mots = re.findall(r"[\w'’À-ÿ-]+", phrase)
    if tete:
        while mots and mots[0].lower() in _TETE:
            mots.pop(0)
    return " ".join(mots[:max_mots]).strip()


_SECTION = re.compile(
    r"^===\s*(?:Définitions?(?: \d+)?|Nom commun(?: \d+)?|"
    r"Nom propre(?: \d+)?|Adjectif(?: qualificatif)?|Verbe|Adverbe)\s*===",
    re.MULTILINE)


def _sens_du_texte(extrait: str) -> str:
    """Premiere definition d'un extrait Wiktionary (plaintext).

    Saute l'entete de langue, l'etymologie et les lignes de prononciation
    (backslash) pour ne garder que les phrases de definition.
    """
    m = _SECTION.search(extrait)
    bloc = extrait[m.end():] if m else extrait
    fin = re.search(r"^(?:===|----)", bloc, re.MULTILINE)
    if fin:
        bloc = bloc[:fin.start()]
    morceaux: list[str] = []
    for ligne in bloc.splitlines():
        morceau = ligne.strip().lstrip("#").strip()
        if not morceau or "\\" in morceau \
                or morceau.startswith("{{") or morceau.startswith("'''"):
            continue                        # prononciation, glose, modele
        morceaux.append(morceau)
    texte = " ".join(morceaux)
    if len(texte) > 420:
        coupe = texte[:420]
        point = max(coupe.rfind(". "), coupe.rfind("! "))
        texte = coupe[:point + 1] if point > 120 else coupe + "…"
    return texte


def _definitions(question: str) -> str:
    m = _MOTIF_DEFI.search(question)
    if not m:
        return ""
    sujet = _mot_recherche(question[m.end():])
    if len(sujet) < 2:
        return ""
    donnee = _json("https://fr.wiktionary.org/w/api.php?"
                   + urllib.parse.urlencode({
                       "action": "query", "prop": "extracts",
                       "exchars": 1400, "explaintext": 1, "redirects": 1,
                       "titles": sujet, "format": "json"}))
    if not donnee or not isinstance(donnee, dict):
        return ""
    pages = donnee.get("query", {}).get("pages") or {}
    if not pages:
        return ""
    page = next(iter(pages.values()))
    if not isinstance(page, dict) or "missing" in page:
        return ""
    corps = _sens_du_texte(str(page.get("extract") or ""))
    if not corps:
        return ""
    titre = str(page.get("title") or sujet)
    return f"Définition de « {titre} » : {corps}"


# ── 4. Open Library : livres et auteurs ───────────────────────────────────

_MOTIF_LIVRE = re.compile(
    r"qui a (?:écrit|ecrit)|auteur\s+(?:de|du|des|d'|d’)"
    r"|quel(?:s)? (?:livre|roman)s?\b|titre (?:du|de|des) (?:livre|roman)"
    r"|livre (?:sur|qui parle de)",
    re.IGNORECASE)


def _livres(question: str) -> str:
    m = _MOTIF_LIVRE.search(question)
    if not m:
        return ""
    sujet = _mot_recherche(question[m.end():], max_mots=8, tete=False)
    if len(sujet) < 2:
        return ""
    donnee = _json("https://openlibrary.org/search.json?"
                   + urllib.parse.urlencode({
                       "q": sujet, "limit": 1,
                       "fields": "title,author_name,first_publish_year"}))
    if not donnee or not isinstance(donnee, dict):
        return ""
    docs = donnee.get("docs") or []
    if not docs or not isinstance(docs[0], dict):
        return ""
    doc = docs[0]
    titre = str(doc.get("title") or "")
    if not titre:
        return ""
    auteurs = ", ".join(str(a) for a in (doc.get("author_name") or [])[:2])
    annee = doc.get("first_publish_year")
    ligne = f"Livre : « {titre} »"
    if auteurs:
        ligne += f" — {auteurs}"
    if annee:
        ligne += f" ({annee})"
    return ligne + "."


# ── 5. Nominatim : localisation OpenStreetMap ─────────────────────────────

# sans IGNORECASE : « Ou est la capitale » ne doit PAS declencher un
# geocodage, seulement une vraie ville capitalisee (« Ou est Tokyo ? »)
_GEO_ICI = re.compile(
    r"(?:[Oo][ùu] se (?:trouve|situe|trouver)|coordonn[ée]es|latitude|"
    r"longitude|localisation|gps)\s+(?:de|du|des|d'|d’)?\s*(.+)")
_GEO_OU = re.compile(
    r"[Oo][Uù] est\s+([A-ZÀ-ÞŒ][\w'’-]+(?:\s+[A-ZÀ-ÞŒ][\w'’-]+)?)")


def _geo(question: str) -> str:
    m = _GEO_ICI.search(question) or _GEO_OU.search(question)
    if not m:
        return ""
    lieu = re.split(r"[?!.]", m.group(1))[0].strip(" «»\"'").strip()
    if len(lieu) < 2:
        return ""
    donnee = _json("https://nominatim.openstreetmap.org/search?"
                   + urllib.parse.urlencode({
                       "q": lieu, "format": "jsonv2",
                       "limit": 1, "accept-language": "fr"}))
    if not donnee or not isinstance(donnee, list) or not donnee:
        return ""
    fiche = donnee[0]
    if not isinstance(fiche, dict):
        return ""
    nom = str(fiche.get("display_name") or lieu)[:160]
    return f"Localisation : {nom} ({_g(fiche.get('lat'))}, {_g(fiche.get('lon'))})."


# ── registre ──────────────────────────────────────────────────────────────

class _Connecteur(NamedTuple):
    nom: str
    motifs: tuple[re.Pattern[str], ...]
    fonction: Callable[[str], str]


_CONNECTEURS: tuple[_Connecteur, ...] = (
    _Connecteur("meteo", (_MOTIF_METEO,), _meteo),
    _Connecteur("devises", (_CONV1, _CONV2), _devises),
    _Connecteur("wiktionary", (_MOTIF_DEFI,), _definitions),
    _Connecteur("openlibrary", (_MOTIF_LIVRE,), _livres),
    _Connecteur("nominatim", (_GEO_ICI, _GEO_OU), _geo),
)


def detecte(question: str) -> bool:
    """Vrai si UNE API de la liste peut repondre a cette question.

    Regex locale uniquement : aucun octet n'est envoye sinon (cout nul
    pour le banc, les puzzles, le calcul verbal...).
    """
    if not _actives():
        return False
    return any(motif.search(question)
               for connecteur in _CONNECTEURS
               for motif in connecteur.motifs)


def chercher(question: str) -> str:
    """Contexte des API correspondantes, vide si aucune/echec/reseau off.

    Les connecteurs concernes s'enchainnent (rarement plus de deux) ; la
    tache est lancee en parallele du web/PGS par l'orchestrateur.
    """
    if not _actives():
        return ""
    morceaux: list[str] = []
    for connecteur in _CONNECTEURS:
        if not any(motif.search(question) for motif in connecteur.motifs):
            continue
        try:
            texte = connecteur.fonction(question)
        except Exception as e:              # jamais bloquant
            LOG.info("[api %s] echec : %s", connecteur.nom, e)
            continue
        if texte:
            LOG.info("[api %s] %d car.", connecteur.nom, len(texte))
            morceaux.append(texte)
    return "\n".join(morceaux)
