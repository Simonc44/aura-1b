"""PAL (Program-Aided) : dates, unites et pourcentages composes — exact, < 1 ms.

Inspiré de PAL/Program-of-Thoughts : sur un probleme chiffre, ne JAMAIS
laisser le LLM deviner — le programme calcule. Ce module etend le niveau 0
aux questions du quotidien que `_calcul_direct` ne couvre pas :

1. **Pourcentages composes** — « 15% de 200 plus 30% de 100 » : les motifs
   « X% de Y » sont resolus d'abord, l'expression restante est evaluee par
   l'evaluateur AST garde du niveau 0 (une seule surface d'attaque).
2. **Unites** — « 5 miles en km », « 100 f en c », « 2 go en mo » : table
   de facteurs purement locale (pas de taux de change : il bouge, on ne
   devine pas). Go/Mo/ko en binaire (1024), comme l'affiche Windows.
3. **Dates** — « date dans 45 jours », « combien de jours jusqu'au
   25 decembre », « quel jour etait le 14 juillet 1789 » : module datetime,
   rien de plus.

Tout est deterministe : une reponse de PAL ne peut pas etre fausse, elle
peut au pire ne pas etre produite (None = le LLM prend la main).
"""
import ast
import calendar
import re
import unicodedata
from datetime import date, datetime, timedelta

from . import filtre_instantane as _f0

# ------------------------------------------------------------------
# Normalisation (meme traitement que le niveau 0)
# ------------------------------------------------------------------

def _normaliser(question: str) -> str:
    q = unicodedata.normalize("NFKD", question.lower())
    return "".join(c for c in q if not unicodedata.combining(c))


def _evaluer_expression(expr: str) -> float:
    """Evalue une expression arithmetique via l'evaluateur GARDE du niveau 0.

    Import paresseux pour eviter le cycle pal -> filtre_instantane -> pal :
    filtre_instantane importe pal (branche du niveau 0), pal ne l'importe
    qu'au moment d'evaluer.
    """
    return _f0._evaluer(ast.parse(expr, mode="eval"))


# ------------------------------------------------------------------
# 1. Pourcentages composes
# ------------------------------------------------------------------

_POURCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:%|pour\s?cent)\s*(?:de|du|des)\s*(\d+(?:[.,]\d+)?)")


def _pourcentages(q: str) -> str | None:
    """« 15% de 200 plus 30% de 100 » -> 60. None si rien a faire."""
    m = _POURCENT.search(q)
    if not m:
        return None
    expr = _POURCENT.sub(lambda m: f"(({m.group(1).replace(',', '.')}/100)*{m.group(2).replace(',', '.')})", q)
    # l'expression restante (mots convertis ou non) passe dans le moule
    # commun : operateurs en mots -> symboles, puis AST garde.
    expr_mots = _f0._expression_en_mots(expr)
    if expr_mots is None:
        # un seul pourcentage isole : « 15% de 200 » -> l'expression
        # residuelle est deja de l'arithmetique pure
        residu = " ".join("".join(_f0._RESIDUS.findall(expr)).split())
        if len(_f0._NOMBRE.findall(residu)) >= 1 and "(" in residu:
            expr_mots = residu
        else:
            return None
    try:
        return _f0._fmt(_evaluer_expression(expr_mots))
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


# ------------------------------------------------------------------
# 2. Unites
# ------------------------------------------------------------------

# (dimension, facteur vers la base) — Go/Mo/ko en binaire (1024), comme
# l'affichage Windows ; an = 365.25 jours.
_UNITES: dict[str, tuple[str, float]] = {
    # longueur (base : metre)
    "km": ("longueur", 1000.0), "kilometre": ("longueur", 1000.0),
    "m": ("longueur", 1.0), "metre": ("longueur", 1.0),
    "cm": ("longueur", 0.01), "mm": ("longueur", 0.001),
    "mile": ("longueur", 1609.344), "miles": ("longueur", 1609.344),
    "pied": ("longueur", 0.3048), "pieds": ("longueur", 0.3048),
    "ft": ("longueur", 0.3048),
    "pouce": ("longueur", 0.0254), "pouces": ("longueur", 0.0254),
    "inch": ("longueur", 0.0254), "inches": ("longueur", 0.0254),
    "yard": ("longueur", 0.9144), "yards": ("longueur", 0.9144),
    # masse (base : kilogramme)
    "kg": ("masse", 1.0), "kilogramme": ("masse", 1.0),
    "g": ("masse", 0.001), "gramme": ("masse", 0.001),
    "mg": ("masse", 1e-6),
    "tonne": ("masse", 1000.0), "tonnes": ("masse", 1000.0),
    "livre": ("masse", 0.45359237), "livres": ("masse", 0.45359237),
    "lbs": ("masse", 0.45359237), "lb": ("masse", 0.45359237),
    "once": ("masse", 0.0283495), "onces": ("masse", 0.0283495),
    "oz": ("masse", 0.0283495),
    # volume (base : litre)
    "l": ("volume", 1.0), "litre": ("volume", 1.0), "litres": ("volume", 1.0),
    "ml": ("volume", 0.001), "cl": ("volume", 0.01),
    "gallon": ("volume", 3.785411784), "gallons": ("volume", 3.785411784),
    # donnees (base : octet, binaire 1024 comme Windows)
    "ko": ("donnees", 1024.0), "mo": ("donnees", 1024.0 ** 2),
    "go": ("donnees", 1024.0 ** 3), "to": ("donnees", 1024.0 ** 4),
    "ko_": ("donnees", 1024.0),
    # temps (base : seconde)
    "ms": ("temps", 0.001),
    "seconde": ("temps", 1.0), "secondes": ("temps", 1.0),
    "min": ("temps", 60.0), "minute": ("temps", 60.0), "minutes": ("temps", 60.0),
    "heure": ("temps", 3600.0), "heures": ("temps", 3600.0),
    "jour": ("temps", 86400.0), "jours": ("temps", 86400.0),
    "semaine": ("temps", 604800.0), "semaines": ("temps", 604800.0),
    "an": ("temps", 31557600.0), "ans": ("temps", 31557600.0),
    "annee": ("temps", 31557600.0),
}
# temperature : formules speciales (offset), pas un simple facteur
_TEMPERATURES = {"c": "c", "celsius": "c", "f": "f", "fahrenheit": "f",
                 "k": "k", "kelvin": "k"}

# les token courts (m, g, l) ne matchent que separes : \b ne suffit pas
# (« 3 g de farine » ok, mais « 3 g » dans « 3 gb » non — gb n'existe pas
# ici donc aucun risque). Les tokens longs sont sans ambiguited.
_MOT_UNITE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(" + "|".join(sorted(_UNITES, key=len, reverse=True)) + r")\b")

# cibles demandees : « en km », « vers km », « combien de km font 5 miles »
# (pluriel accepte apres le nom d'unite, hors capture)
_CIBLE = re.compile(
    r"(?:\b(?:en|vers|to|in)\s+|\bcombien\s+de\s+|\bc'est\s+combien\s+de\s+)"
    r"(" + "|".join(sorted(_UNITES, key=len, reverse=True)) + r")(?:s|es)?\b"
    r"|(" + "|".join(_TEMPERATURES) + r")\b")


def _convertir_temperature(v: float, src: str, dst: str) -> float | None:
    if src == dst:
        return v
    if src == "c" and dst == "f":
        return v * 9 / 5 + 32
    if src == "f" and dst == "c":
        return (v - 32) * 5 / 9
    if src == "c" and dst == "k":
        return v + 273.15
    if src == "k" and dst == "c":
        return v - 273.15
    if src == "f" and dst == "k":
        return (v - 32) * 5 / 9 + 273.15
    if src == "k" and dst == "f":
        return (v - 273.15) * 9 / 5 + 32
    return None


def _unites(q: str) -> str | None:
    """« 5 miles en km » -> « 5 miles = 8.04672 km ». None sinon."""
    # temperature d'abord : les tokens f/c/k ne sont pas dans _UNITES,
    # donc la source doit etre lue par une regex dediee
    temps_tries = sorted(_TEMPERATURES, key=len, reverse=True)
    m_src = re.search(
        rf"(-?\d+(?:[.,]\d+)?)\s*(?:degres?\s+)?({'|'.join(temps_tries)})\b", q)
    if m_src and m_src.group(2) in _TEMPERATURES:
        m_dst = re.search(
            rf"\b(?:en|vers|to|in|combien\s+de)\s+({'|'.join(temps_tries)})\b", q)
        if m_dst:
            valeur = float(m_src.group(1).replace(",", "."))
            r = _convertir_temperature(valeur, _TEMPERATURES[m_src.group(2)],
                                       _TEMPERATURES[m_dst.group(1)])
            if r is not None:
                return (f"{_f0._fmt(valeur)} {m_src.group(2)} = "
                        f"{_f0._fmt(r)} {m_dst.group(1)}")
            return None

    src = _MOT_UNITE.search(q)
    if not src:
        return None
    valeur = float(src.group(1).replace(",", "."))
    unite_src = src.group(2)

    cible = _CIBLE.search(q)
    if not cible:
        return None
    unite_dst = cible.group(1) or cible.group(2)
    if unite_dst not in _UNITES:
        return None
    dim_src, fac_src = _UNITES[unite_src]
    dim_dst, fac_dst = _UNITES[unite_dst]
    if dim_src != dim_dst:
        return None
    r = valeur * fac_src / fac_dst
    return f"{_f0._fmt(valeur)} {unite_src} = {_f0._fmt(r)} {unite_dst}"


# ------------------------------------------------------------------
# 3. Dates
# ------------------------------------------------------------------

_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MOIS = ("janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet",
         "aout", "septembre", "octobre", "novembre", "decembre")
_MOIS_ALT = {"juil": "juillet", "fev": "fevrier", "avr": "avril", "sept": "septembre", "dec": "decembre", "janv": "janvier", "nov": "novembre", "oct": "octobre"}

_RE_MOIS = "|".join(_MOIS) + "|" + "|".join(_MOIS_ALT)


def _aujourdhui() -> date:
    """Point d'injection pour les tests (monkeypatch)."""
    return date.today()


def _lire_date(q: str) -> date | None:
    """Lit une date française dans le texte : « 25/12/2026 », « 25 decembre
    2026 », « 1er mai ». Annee absente -> annee courante."""
    m = re.search(rf"\b(\d{{1,2}})(?:er)?\s+({_RE_MOIS})\b(?:\s*(\d{{4}}))?", q)
    if m:
        jour = int(m.group(1))
        mois_txt = _MOIS_ALT.get(m.group(2), m.group(2))
        mois = _MOIS.index(mois_txt) + 1
        annee = int(m.group(3)) if m.group(3) else _aujourdhui().year
        try:
            return date(annee, mois, jour)
        except ValueError:
            return None
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", q)
    if m:
        jour, mois = int(m.group(1)), int(m.group(2))
        annee = int(m.group(3)) if m.group(3) else _aujourdhui().year
        if annee < 100:
            annee += 2000
        try:
            return date(annee, mois, jour)
        except ValueError:
            return None
    return None


def _ajouter_mois(d: date, n: int) -> date:
    """Ajoute n mois calendaire (30/31 fevrier etc. geres par le calendrier)."""
    total = d.month - 1 + n
    annee = d.year + total // 12
    mois = total % 12 + 1
    jour = min(d.day, calendar.monthrange(annee, mois)[1])
    return date(annee, mois, jour)


def _fmt_date(d: date) -> str:
    return f"{_JOURS[d.weekday()]} {d.day} {_MOIS[d.month - 1]} {d.year}"


def _prochaine_occurrence(d: date) -> date:
    """Date sans annee deja passee -> anniversaire/epoque annuelle : +1 an."""
    auj = _aujourdhui()
    if d < auj:
        try:
            return d.replace(year=auj.year + 1)
        except ValueError:          # 29 fevrier
            return d.replace(year=auj.year + 1, day=28)
    return d


def _dates(q: str) -> str | None:
    """Resout les questions de dates courantes. None sinon."""
    auj = _aujourdhui()

    # « dans 45 jours », « dans 3 semaines », « dans 2 mois »
    m = re.search(
        r"\bdans\s+(\d+)\s+(minute|heure|jour|semaine|mois|an|annee)s?\b", q)
    if m:
        n, unite = int(m.group(1)), m.group(2)
        if unite == "mois":
            echeance = _ajouter_mois(auj, n)
        elif unite in ("an", "annee"):
            echeance = _ajouter_mois(auj, 12 * n)
        else:
            echeance = auj + {"minute": timedelta(minutes=n),
                              "heure": timedelta(hours=n),
                              "jour": timedelta(days=n),
                              "semaine": timedelta(weeks=n)}[unite]
        return f"{_fmt_date(echeance)}"

    # « N jours apres le DATE » / « N jours avant le DATE »
    m = re.search(r"\b(\d+)\s+(jour|semaine|mois)s?\s+(apres|avant)\b", q)
    if m:
        base = _lire_date(q)
        if base:
            n = int(m.group(1))
            sens = 1 if m.group(3) == "apres" else -1
            if m.group(2) == "semaine":
                n *= 7
            elif m.group(2) == "mois":
                echeance = _ajouter_mois(base, sens * n)
                return _fmt_date(echeance)
            return _fmt_date(base + timedelta(days=sens * n))

    # « combien de jours entre le X et le Y »
    m = re.search(r"\bcombien\s+de\s+jours\s+entre\b", q)
    if m:
        morceaux = re.findall(_RE_DATE_EXTRACT, q)
        if len(morceaux) >= 2:
            d1, d2 = _lire_date(morceaux[0]), _lire_date(morceaux[1])
            if d1 and d2:
                return f"{abs((d2 - d1).days)} jours"
        return None

    # « combien de jours jusqu'au DATE / jusqu au / avant le / d'ici DATE »
    m = re.search(
        r"\bcombien\s+de\s+jours\s+(?:jusqu[' ]?a(?:u)?\b|d[' ]?ici\b|avant\s+le\b)", q)
    if m:
        jusqu: date | None = _lire_date(q)
        if jusqu:
            jusqu = _prochaine_occurrence(jusqu)
            return f"{(jusqu - auj).days} jours"
        return None

    # « combien de jours depuis le DATE »
    m = re.search(r"\bcombien\s+de\s+jours\s+depuis\b", q)
    if m:
        depuis: date | None = _lire_date(q)
        if depuis:
            delta = (auj - depuis).days
            if delta >= 0:
                return f"{delta} jours"
            return f"dans {-delta} jours"
        return None

    # « quel jour etait/sera le DATE »
    m = re.search(r"\bquel\s+jour\s+(?:etait|sera|est|tombe)\b", q)
    if m:
        jour_cible: date | None = _lire_date(q)
        if jour_cible:
            return _JOURS[jour_cible.weekday()]
        return None

    # « date de demain / hier / apres-demain »
    m = re.search(r"\bdate\s+(?:de\s+)?(demain|hier|apres-demain)\b", q)
    if m:
        dec = {"hier": -1, "demain": 1, "apres-demain": 2}[m.group(1)]
        return _fmt_date(auj + timedelta(days=dec))
    return None


# extraction de fragment contenant une date (pour « entre X et Y »)
_RE_DATE_EXTRACT = re.compile(
    rf"\b(?:le\s+)?\d{{1,2}}(?:er)?\s+(?:{_RE_MOIS})\b(?:\s+\d{{4}})?"
    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?")


# ------------------------------------------------------------------
# Point d'entree (branche du niveau 0)
# ------------------------------------------------------------------

def repondre(question: str) -> str | None:
    """Tente une reponse PAL exacte. None = rien pour moi, LLM prend la main."""
    q = _normaliser(question)
    r = _pourcentages(q)
    if r is not None:
        return r
    r = _unites(q)
    if r is not None:
        return r
    return _dates(q)
