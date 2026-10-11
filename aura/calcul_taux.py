"""Calcul de taux : une unite par duree (produit) ou un total par duree (division).

Le 1B ne sait pas CALCULER un probleme raconte a taux : il ENONCE la
formule. Mesure au banc 100 (scripts/banc_100.py) — 5 echecs, tous de la
meme famille :

  « Un train roule a 80 km/h pendant 3 heures. Quelle distance parcourt-il ? »
  -> rendait « Pour trouver la distance parcourue par le train, on peut
     utiliser la formule : ... »   (formule citee, aucun calcul) ; attendu 240.
  « Une voiture parcourt 150 km en 3 heures. Quelle est sa vitesse moyenne ? »
  -> meme motif ; attendu 50.
  « Un ouvrier gagne 15 euros de l'heure. Il travaille 8 heures. »
  -> meme motif ; attendu 120.
  « Une pizza de 8 parts est partagee entre 4 personnes. »
  -> meme motif ; attendu 2.
  « Un robinet remplit 12 litres par minute. Combien de litres en 5 minutes ? »
  -> meme motif ; attendu 60.

C'est le mode d'echec documente des petits modeles : une chaine de
raisonnement FLUENTE mais FAUSSE (imitation de la forme, pas de logique).
Le remede n'est pas un meilleur prompt mais un calculateur.

Trois formes reconnues, une seule operation chacune :
  A) taux x duree      : « 80 km/h pendant 3 heures »   -> 240
                         « 12 litres par minute en 5 minutes » -> 60
                         « 15 euros de l'heure, 8 heures »     -> 120
  B) total / duree     : « 150 km en 3 heures » + question de VITESSE -> 50
  C) total / parts     : « 8 parts partagees entre 4 »  -> 2

Discipline identique a calcul_achat et calcul_verbal : toute ambiguite
-> None (cascade vers le cerveau, zero risque) :
  - aucune conversion d'unite de temps (heures vs minutes) : on s'abstient ;
  - deux formes a la fois qui ne donnent pas le meme resultat : on s'abstient ;
  - sans point d'interrogation ni mot interrogatif : ce n'est pas une question.

Position dans la cascade (orchestrateur) : APRES calcul_verbal — mesure
faite, calcul_verbal ne repond JAMAIS sur ces 5 questions (il rend None),
donc ce module ne peut rien lui retirer et n'ajoute que du rappel.
"""
import re


def _sans_accents(texte: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in n if not unicodedata.combining(c))


def _fmt(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v):,}".replace(",", " ")
    return f"{v:.6g}"


# Le probleme doit POSER une question : sans cela, rien a repondre.
_RE_QUESTION = re.compile(r"\?|\b(?:combien|quel(?:le)?)\b")

# A) Un TAUX, deux ecritures.
#    « 80 km/h », « 12 litres par minute », « 3 m/s »
_RE_TAUX_BARRE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[a-z]*\s*(?:/|par)\s*"
    r"(heure|heures|h|minute|minutes|min|seconde|secondes|s|jour|jours)\b")
#    « 15 euros de l'heure », « 20 euros par jour »
#    La DEVISE est obligatoire : sans elle, « 3 heures » se lirait comme un
#    taux de 3 par heure et « 150 km en 3 heures » donnerait 9 au lieu de
#    rien du tout. On ne devine pas un taux qui n'est pas ecrit.
_RE_TAUX_PRIX = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:euros?|euro|dollars?|dollar|€)\s*"
    r"(?:de\s+l'?|de\s+la|par\s+l'?|par\s+la|a\s+l'?|a\s+la|l'|/)\s*"
    r"(heure|heures|h|minute|minutes|min|seconde|secondes|s|jour|jours)\b")

# La DUREE sur laquelle on applique le taux.
_RE_DUREE = re.compile(
    r"(?:pendant|durant|en|sur|travaille|dure|met)\s+"
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(heure|heures|h|minute|minutes|min|seconde|secondes|s|jour|jours)\b")

# B) Un TOTAL avec une unite de quantite, pour une question de vitesse/debit.
_RE_TOTAL = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(km|kilometres?|metres?|miles?|litres?|l\b)\b")
_RE_DEMANDE_TAUX = re.compile(r"\b(?:vitesse|debit)\b")

# C) Un PARTAGE : « 8 parts partagees entre 4 personnes ».
_RE_PARTAGE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s+[a-z]+\s*"
    r"(?:est|sont)?\s*"
    r"(?:partage|reparti|distribu|divise)[a-z]*\s*"
    r"(?:entre|sur|par)\s+"
    r"(\d+(?:[.,]\d+)?)\s+[a-z]+")

# Unites de temps ramenees a une forme canonique. Pas de conversion entre
# elles : « par minute » + « en heures » s'abstient (voir en-tete).
_TEMPS = {
    "heure": "h", "heures": "h", "h": "h",
    "minute": "min", "minutes": "min", "min": "min",
    "seconde": "s", "secondes": "s", "s": "s",
    "jour": "j", "jours": "j",
}


def _nombre(brut: str) -> float:
    return float(brut.replace(",", "."))


def resoudre(question: str) -> str | None:
    """Le resultat d'un probleme a taux, ou None si ambigu."""
    q = _sans_accents(question)
    if not _RE_QUESTION.search(q):
        return None

    resultats: set[str] = set()

    # -- A) taux x duree -------------------------------------------------
    # La question ne doit pas DEMANDER un taux : sinon c'est la forme B.
    if not _RE_DEMANDE_TAUX.search(q):
        mt = _RE_TAUX_BARRE.search(q)
        if mt is None:
            mt = _RE_TAUX_PRIX.search(q)
        md = _RE_DUREE.search(q)
        if mt and md and _TEMPS[mt.group(2)] == _TEMPS[md.group(2)]:
            resultats.add(_fmt(_nombre(mt.group(1)) * _nombre(md.group(1))))

    # -- B) total / duree (question de vitesse ou de debit) --------------
    if _RE_DEMANDE_TAUX.search(q):
        md = _RE_DUREE.search(q)
        totaux = _RE_TOTAL.findall(q)
        # Un seul total clair : plusieurs chiffres = ambiguite -> on s'abstient.
        if md and len(totaux) == 1 and _TEMPS[md.group(2)] == "h":
            resultats.add(_fmt(_nombre(totaux[0][0]) / _nombre(md.group(1))))

    # -- C) partage ------------------------------------------------------
    mp = _RE_PARTAGE.search(q)
    if mp and _nombre(mp.group(2)) != 0:
        resultats.add(_fmt(_nombre(mp.group(1)) / _nombre(mp.group(2))))

    # Une seule lecture possible : sinon on ne devine pas.
    if len(resultats) != 1:
        return None
    return resultats.pop()


def verifier(question: str, reponse: str) -> bool:
    """La reponse doit etre un nombre et la question poser un taux."""
    q = _sans_accents(question)
    if not reponse or not _RE_QUESTION.search(q):
        return False
    return bool(_RE_TAUX_BARRE.search(q) or _RE_TAUX_PRIX.search(q)
                or _RE_PARTAGE.search(q))
