"""Banc 100 questions — mesure LARGE, complement de bench_qualite.py.

Difference avec bench_qualite.py (14 questions) : celui-ci est un garde-fou
de non-regression, celui-la mesure des TAUX. Le volume permet de voir quelle
categorie est faible, pas seulement qu'une question a regresse.

Le cache semantique et la memoire conversationnelle sont isoles par run;
les reponses du banc ne sont pas reinjectees dans les questions suivantes.

DEUX FAMILLES DE CONTROLES — la distinction est affichee dans le rapport :
  * VERIFIABLE — la bonne reponse est connue (nombre, ou mot unique).
    Le controle est STRICT : la valeur doit etre presente, sinon 0.
    Categories : exact, verbal, raisonnement, logique, factuel, format.
  * FORME — le banc ne peut pas prouver la verite d'une dissertation.
    Le controle verifie la bonne formation : pas de boucle, pas de balise
    de plan recopiee, pas de refus en tete, longueur credible.
    Categories : riche, meta.

Usage :
    .venv/Scripts/python.exe scripts/banc_100.py
    .venv/Scripts/python.exe scripts/banc_100.py --rapide     # sans les riches
    .venv/Scripts/python.exe scripts/banc_100.py --cat exact
    .venv/Scripts/python.exe scripts/banc_100.py --json banc_100.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RACINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_qualite as bq  # noqa: E402  (pose AURA_FILTRE_CACHE au passage)

_controle_riche = bq._controle_riche
_sans_balise = bq._sans_balise
_sans_boucle = bq._sans_boucle

# ── extraction de nombres ────────────────────────────────────────────────
# On teste l'APPARTENANCE d'un entier et non la sous-chaine : « 12 » est une
# sous-chaine de « 120 », donc un mauvais resultat (120 pour 12) passerait.
_RE_ENTIER = re.compile(r"\d+")

# Nombres en toutes lettres : le 1B repond parfois « neuf » ou « quatre ».
_MOTS_NOMBRES = {
    "zero": 0, "un": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11,
    "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
    "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60, "cent": 100,
}


def _entiers(texte: str) -> set[int]:
    """Entiers presents dans le texte (separateurs de milliers inclus)."""
    t = re.sub(r"(?<=\d)[\s\u00a0\u202f](?=\d)", "", texte)
    return {int(m) for m in _RE_ENTIER.findall(t)}


def _a_nombre(reponse: str, attendu: int) -> bool:
    """La reponse contient-elle EXACTEMENT ce nombre (chiffres ou lettres) ?"""
    if attendu in _entiers(reponse):
        return True
    bas = re.sub(r"[\s\u00a0\u202f]+", " ", reponse.lower())
    return any(re.search(rf"\b{m}\b", bas)
               for m, v in _MOTS_NOMBRES.items() if v == attendu)


def _nombre(attendu: int, latence_max: float | None = None):
    """Controle strict : le nombre attendu, et rien d'autre a verifier."""
    def controle(reponse: str, meta: dict) -> tuple[bool, str]:
        mesure = f"{meta['latence']:.1f}s, {len(reponse)} car"
        if latence_max is not None and meta["latence"] > latence_max:
            return False, mesure + f"  KO:lent(>{latence_max:.0f}s)"
        if not _a_nombre(reponse, attendu):
            return False, mesure + f"  KO:attendu={attendu}"
        return True, mesure
    return controle


def _mots(*morceaux: str, latence_max: float | None = None):
    """Controle strict : tous les mots attendus sont presents (insensible casse)."""
    def controle(reponse: str, meta: dict) -> tuple[bool, str]:
        bas = re.sub(r"\s+", " ", reponse.lower())
        manquants = [m for m in morceaux if m.lower() not in bas]
        mesure = f"{meta['latence']:.1f}s, {len(reponse)} car"
        if latence_max is not None and meta["latence"] > latence_max:
            return False, mesure + f"  KO:lent(>{latence_max:.0f}s)"
        if manquants:
            return False, mesure + "  KO:absent=" + ",".join(manquants)
        return True, mesure
    return controle


def _plie(s: str) -> str:
    """Minuscules SANS accents : « lumière » et « lumiere » se comparent."""
    import unicodedata
    n = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"\s+", " ", "".join(c for c in n
                                         if not unicodedata.combining(c)))


def _mots_ou(*morceaux: str, latence_max: float | None = None):
    """Controle strict : AU MOINS UN des mots attendus est present.

    Pour les questions a plusieurs formulations justes (« lumiere du
    Soleil » / « chlorophylle ») : exiger un mot precis transformait une
    bonne reponse en KO (constate au premier run).
    """
    def controle(reponse: str, meta: dict) -> tuple[bool, str]:
        bas = _plie(reponse)
        mesure = f"{meta['latence']:.1f}s, {len(reponse)} car"
        if latence_max is not None and meta["latence"] > latence_max:
            return False, mesure + f"  KO:lent(>{latence_max:.0f}s)"
        if not any(_plie(m) in bas for m in morceaux):
            return False, mesure + "  KO:aucun de " + "|".join(morceaux)
        return True, mesure
    return controle


def _forme(latence_max: float | None = None, mini: int = 40):
    """Controle de FORME : pas de boucle, pas de balise, longueur credible."""
    def controle(reponse: str, meta: dict) -> tuple[bool, str]:
        defauts = []
        if not _sans_boucle(reponse):
            defauts.append("boucle")
        if not _sans_balise(reponse):
            defauts.append("balise")
        if len(reponse) < mini:
            defauts.append("court")
        if latence_max is not None and meta["latence"] > latence_max:
            defauts.append(f"lent(>{latence_max:.0f}s)")
        mesure = f"{meta['latence']:.1f}s, {len(reponse)} car"
        if defauts:
            mesure += "  KO:" + ",".join(defauts)
        return (not defauts), mesure
    return controle


# ── les 100 questions ────────────────────────────────────────────────────
# (id, categorie, question, controle)
QUESTIONS: list[tuple[str, str, str, object]] = []


def _q(qid, cat, question, controle):
    QUESTIONS.append((qid, cat, question, controle))


# --- exact : 30, le pilier « the fastest answer is the one you never generate »
_EXACTS = [
    (391, "Combien font 17 x 23 ?"),
    (169, "Quel est le carre de 13 ?"),
    (12, "Combien font 144 divise par 12 ?"),
    (1024, "Quel est 2 puissance 10 ?"),
    (56, "Combien font 7 x 8 ?"),
    (146, "Combien font 99 + 47 ?"),
    (622, "Combien font 1000 - 378 ?"),
    (27, "Quel est le cube de 3 ?"),
    (54, "Combien font 6 x 9 ?"),
    (270, "Combien font 45 x 6 ?"),
    (9, "Combien font 81 divise par 9 ?"),
    (36, "Combien font 15 pour cent de 240 ?"),
    (144, "Quel est le carre de 12 ?"),
    (625, "Combien font 250 + 375 ?"),
    (399, "Combien font 19 x 21 ?"),
    (125, "Combien font 1000 divise par 8 ?"),
    (343, "Quel est le cube de 7 ?"),
    (39, "Combien font 68 - 29 ?"),
    (256, "Combien font 16 x 16 ?"),
    (128, "Quel est 2 puissance 7 ?"),
    (30, "Combien font 360 divise par 12 ?"),
    (20, "Combien font 25 pour cent de 80 ?"),
    (121, "Combien font 11 x 11 ?"),
    (443, "Combien font 900 - 457 ?"),
    (238, "Combien font 14 x 17 ?"),
    (512, "Quel est le cube de 8 ?"),
    (100, "Combien font 27 + 58 + 15 ?"),
    (384, "Combien font 128 + 256 ?"),
    (666, "Combien font 222 x 3 ?"),
    (45, "Combien font 5 x 9 ?"),
]
# Pas de seuil de latence ici : cette categorie mesure l'EXACTITUDE de la
# valeur. Une latence elevee (cache manque -> LLM) est un signal de
# performance, deja rapporte par le tableau des latences par categorie ; la
# melanger au score confondrait « faux » et « lent ».
for _i, (_v, _q_txt) in enumerate(_EXACTS, 1):
    _q(f"exact-{_i:02d}", "exact", _q_txt, _nombre(_v))

# --- verbal : 15, problemes racontes a une etape
_VERBAUX = [
    (7, "J'ai 12 bonbons, j'en donne 5, combien m'en reste-t-il ?"),
    (6, "J'ai 10 euros et je depense 4 euros, combien me reste-t-il ?"),
    (15, "Il y a 8 pommes dans un panier, on en ajoute 7, combien y en a-t-il ?"),
    (15, "Un stylo coute 3 euros, j'en achete 5, combien je paie ?"),
    (12, "J'avais 20 billes, j'en perds 8, combien m'en reste-t-il ?"),
    (5, "Marie a 14 ans et son frere a 9 ans, quel est l'ecart d'age ?"),
    (42, "Il y a 6 rangees de 7 chaises, combien de chaises ?"),
    (15, "Un paquet contient 24 biscuits, j'en mange 9, combien en reste-t-il ?"),
    (8, "J'ai 15 euros et j'achete un livre a 7 euros, combien me reste-t-il ?"),
    (18, "Une boite contient 30 crayons, on en retire 12, combien reste-t-il ?"),
    (24, "Paul a 3 fois 8 billes, combien en a-t-il ?"),
    (36, "Il y a 9 eleves et chacun a 4 cahiers, combien de cahiers ?"),
    (27, "J'ai 50 cartes, j'en distribue 23, combien m'en reste-t-il ?"),
    (7, "Un gateau a 12 parts, on en coupe 5, combien reste-t-il ?"),
    (27, "Un cycliste roule 18 km puis 9 km, combien en tout ?"),
]
for _i, (_v, _q_txt) in enumerate(_VERBAUX, 1):
    _q(f"verbal-{_i:02d}", "verbal", _q_txt, _nombre(_v))

# --- raisonnement : 15, plusieurs etapes (le terrain du PoT)
_RAISONNEMENTS = [
    (8, "Un cahier coute 3 euros. J'en achete 4 et je paie avec un billet "
        "de 20 euros. Combien me rend-on ?"),
    (240, "Un train roule a 80 km/h pendant 3 heures. Quelle distance "
          "parcourt-il ?"),
    (16, "J'achete 3 stylos a 2 euros et 2 cahiers a 5 euros. Combien je "
         "paie en tout ?"),
    (4, "Alice a 12 bonbons, elle en donne la moitie a Bob puis 2 a Clara. "
        "Combien lui en reste-t-il ?"),
    (4, "Un magasin vend des pommes a 2 euros le kilo. J'en achete 3 kilos "
        "et je paie avec 10 euros. Combien me rend-on ?"),
    (50, "Une voiture parcourt 150 km en 3 heures. Quelle est sa vitesse "
         "moyenne en km/h ?"),
    (53, "Il y a 5 boites de 12 oeufs. J'en casse 7. Combien d'oeufs entiers "
         "reste-t-il ?"),
    (120, "Un ouvrier gagne 15 euros de l'heure. Il travaille 8 heures. "
          "Combien gagne-t-il ?"),
    (22, "Un rectangle mesure 7 cm de long et 4 cm de large. Quel est son "
         "perimetre en cm ?"),
    (28, "Un rectangle mesure 7 cm de long et 4 cm de large. Quelle est son "
         "aire en cm carres ?"),
    (46, "J'ai 100 euros. J'achete 3 livres a 18 euros chacun. Combien me "
         "reste-t-il ?"),
    (40, "Un reservoir de 60 litres est rempli aux deux tiers. Combien de "
         "litres contient-il ?"),
    (2, "Une pizza de 8 parts est partagee entre 4 personnes. Combien de "
        "parts chacune ?"),
    (37, "Un bus contient 45 passagers. 17 descendent et 9 montent. Combien "
         "de passagers reste-t-il ?"),
    (60, "Un robinet remplit 12 litres par minute. Combien de litres en "
         "5 minutes ?"),
]
for _i, (_v, _q_txt) in enumerate(_RAISONNEMENTS, 1):
    _q(f"raison-{_i:02d}", "raisonnement", _q_txt, _nombre(_v, latence_max=60.0))

# --- logique : 8, solution unique
_LOGIQUES = [
    (9, "Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que Clara. "
        "Clara a 4 ans. Quel age a Alice ?"),
    (None, "Paul est plus grand que Marc, et Marc est plus grand que Lea. "
           "Qui est le plus grand ?"),
    (None, "Trois amis ont des ages differents : Ana 25 ans, Ben 31 ans, "
           "Chloe 28 ans. Qui est le plus jeune ?"),
    (None, "Le chien est plus lourd que le chat, et le chat est plus lourd "
           "que la souris. Quel animal est le plus lourd ?"),
    (11, "Lea a 2 ans de moins que Tom. Tom a 5 ans de plus que Zoe. "
         "Zoe a 8 ans. Quel age a Lea ?"),
    (None, "Un livre coute plus cher qu'un stylo, et un stylo coute plus "
           "cher qu'une gomme. Quel objet coute le moins cher ?"),
    (11, "Marc a 4 ans de plus que Julie. Julie a 3 ans de moins que Sam. "
         "Sam a 10 ans. Quel age a Marc ?"),
    (None, "Dans une course, A arrive avant B et B arrive avant C. "
           "Qui arrive en dernier ?"),
]
_LOGIQUES_MOTS = {2: "paul", 3: "ana", 4: "chien", 6: "gomme", 8: "c"}
for _i, (_v, _q_txt) in enumerate(_LOGIQUES, 1):
    _ctrl = (_nombre(_v, latence_max=60.0) if _v is not None
             else _mots(_LOGIQUES_MOTS[_i], latence_max=60.0))
    _q(f"logique-{_i:02d}", "logique", _q_txt, _ctrl)

# --- factuel : 12, reponses a valeur unique (verifiables)
_FACTUELS = [
    ("paris", "Quelle est la capitale de la France ?"),
    ("canberra", "Quelle est la capitale de l'Australie ?"),
    ("rome", "Quelle est la capitale de l'Italie ?"),
    ("madrid", "Quelle est la capitale de l'Espagne ?"),
    ("tokyo", "Quelle est la capitale du Japon ?"),
    ("pacifique", "Quel est le plus grand ocean du monde ?"),
    ("mercure", "Quelle est la planete la plus proche du Soleil ?"),
    ("366", "Combien de jours compte une annee bissextile ?"),
    ("7", "Combien y a-t-il de continents sur Terre ?"),
    ("h2o", "Quel est le symbole chimique de l'eau ?"),
    ("6", "Combien de cotes a un hexagone ?"),
]
for _i, (_mot, _q_txt) in enumerate(_FACTUELS, 1):
    _q(f"factuel-{_i:02d}", "factuel", _q_txt, _mots(_mot, latence_max=60.0))
# Question a PLUSIEURS formulations justes : le pigment (chlorophylle) et la
# ressource (lumiere du Soleil) repondent tous deux a la question. Exiger un
# seul mot transformait une bonne reponse en KO.
_q("factuel-12", "factuel",
   "De quoi les plantes ont-elles besoin pour la photosynthese ?",
   _mots_ou("lumiere", "soleil", "chlorophylle", latence_max=60.0))

# --- meta : 5, constantes (0 generation -> latence quasi nulle)
_q("meta-nom", "meta", "Qui es-tu ?",
   lambda r, m: (m["cerveau"].startswith("constante") and "RontoGPT" in r,
                 m["cerveau"]))
_q("meta-createur", "meta", "Qui t'a cree ?",
   lambda r, m: (m["cerveau"].startswith("constante") and "Simon" in r,
                 m["cerveau"]))
_q("meta-nom2", "meta", "Quel est ton nom ?",
   lambda r, m: (m["cerveau"].startswith("constante") and "RontoGPT" in r,
                 m["cerveau"]))
_q("meta-8b", "meta", "Tu es plus fort que ChatGPT ?",
   _forme(latence_max=5.0, mini=10))
_q("meta-8b2", "meta", "es tu une ia aussi puissante qu'un 8B ou plus ?",
   _forme(latence_max=5.0, mini=10))

# --- format : 5, consigne explicite respectee
_q("format-mot", "format", "Ecris exactement le mot BANANE.",
   _mots("banane", latence_max=30.0))
_q("format-couleurs", "format", "Donne trois couleurs primaires separees par "
   "des virgules.",
   lambda r, m: (sum(c in r.lower() for c in ("rouge", "bleu", "jaune")) >= 2,
                 f"{m['latence']:.1f}s"))
_q("format-calcul", "format", "Combien font 4 + 4 ? Reponds en un seul chiffre.",
   _nombre(8, latence_max=30.0))
_q("format-phrase", "format", "Ecris la phrase : je teste RontoGPT.",
   _mots("je teste aura", latence_max=30.0))
_q("format-soustraction", "format", "Donne le resultat de 10 - 3.",
   _nombre(7, latence_max=30.0))

# --- riche : 10, forme seulement (le banc ne peut pas juger la verite)
_RICHES = [
    "Redige une analyse comparee du solaire et de l'eolien en France, avec "
    "les avantages et les limites de chaque filiere",
    "La puissance de la technologie est-elle une menace pour l'homme "
    "moderne ? Redige une dissertation structuree.",
    "Explique pourquoi la lecture est importante pour la liberte de penser, "
    "en trois parties",
    "Analyse les avantages et les inconvenients du teletravail pour les "
    "entreprises et les salaries",
    "Compare la democratie athenienne et la democratie moderne",
    "La liberte peut-elle exister sans contrainte ? Redige une dissertation",
    "Analyse le role de l'art dans la societe contemporaine",
    "Compare l'intelligence artificielle et l'intelligence humaine sur le "
    "plan de la creativite",
    "Explique pourquoi les oceans sont essentiels a l'equilibre climatique",
    "Redige une dissertation structuree sur l'utilite du doute dans la "
    "demarche scientifique",
]
for _i, _q_txt in enumerate(_RICHES, 1):
    _q(f"riche-{_i:02d}", "riche", _q_txt, _controle_riche)

VERIFIABLES = {"exact", "verbal", "raisonnement", "logique", "factuel", "format"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="banc_100")
    ap.add_argument("--rapide", action="store_true",
                    help="saute la categorie riche (la plus lente)")
    ap.add_argument("--cat", metavar="CAT", default="",
                    help="ne lancer qu'une categorie")
    ap.add_argument("--json", metavar="CHEMIN", default="",
                    help="ecrit le detail complet en JSON")
    ap.add_argument("--cache", metavar="CHEMIN", default="",
                    help="cache en lecture seule a evaluer (defaut : cache vide temporaire)")
    args = ap.parse_args(argv)

    cache_temporaire = not args.cache
    chemin_cache = (Path(args.cache) if args.cache else
                    _RACINE / f".banc100_cache_{os.getpid()}_{time.time_ns()}.jsonl")
    chemin_cache.parent.mkdir(parents=True, exist_ok=True)
    chemin_cache.touch(exist_ok=True)
    os.environ["AURA_FILTRE_CACHE"] = str(chemin_cache)

    from aura import config
    config.appliquer()
    from aura.memoire_conversation import MemoireConversation
    from aura import filtre_instantane
    from aura.orchestrateur import RontoGPT

    # Un cache passe explicitement peut fournir des hits, mais le banc ne
    # doit jamais s'y repondre lui-meme en enregistrant ses generations.
    filtre_instantane.enregistrer = lambda question, reponse: None
    filtre_instantane.mettre_a_jour = lambda question, reponse: False

    ia = RontoGPT()
    resultats = []
    t_debut = time.time()
    print(f"{'id':16s} {'categorie':12s} {'ok':>3s}  mesure")
    print("-" * 78)
    for index, (qid, cat, question, controle) in enumerate(QUESTIONS):
        if args.rapide and cat == "riche":
            continue
        if args.cat and cat != args.cat:
            continue
        # Chaque exemple est une question isolee : la memoire personnelle du
        # developpeur et les reponses des questions precedentes ne doivent pas
        # influencer la mesure. Le banc multi-tours dedie couvre la memoire.
        historique = (_RACINE / f".banc100_{os.getpid()}_{index}.jsonl")
        ia._historique = MemoireConversation(chemin=historique)
        reponse = ""
        try:
            reponse, meta = bq._executer(ia, question)
            ok, detail = controle(reponse, meta)          # type: ignore[operator]
        except Exception as e:                             # jamais de crash banc
            ok, detail = False, f"exception: {type(e).__name__}: {e}"
            meta = {"latence": 0.0, "cerveau": "?", "experts": []}
        finally:
            # Le fichier peut contenir la question et la reponse finale.
            for chemin in (historique, historique.with_suffix(".etat.json")):
                try:
                    chemin.unlink()
                except FileNotFoundError:
                    pass
        ligne = {"id": qid, "categorie": cat, "question": question,
                 "ok": bool(ok), "mesure": detail,
                 "verifiable": cat in VERIFIABLES, **meta}
        if not ok and reponse:
            ligne["reponse"] = reponse        # diagnostic du KO (JSON)
        resultats.append(ligne)
        print(f"{qid:16s} {cat:12s} {'OK' if ok else 'KO':>3s}  {detail}",
              flush=True)

    if cache_temporaire:
        try:
            chemin_cache.unlink()
        except FileNotFoundError:
            pass

    total = len(resultats)
    bons = sum(1 for r in resultats if r["ok"])
    ver = [r for r in resultats if r["verifiable"]]
    ver_ok = sum(1 for r in ver if r["ok"])
    forme = [r for r in resultats if not r["verifiable"]]
    forme_ok = sum(1 for r in forme if r["ok"])
    duree = time.time() - t_debut

    print("-" * 78)
    print(f"SCORE GLOBAL      : {bons}/{total}  "
          f"({100 * bons / max(total, 1):.0f} %)   {duree:.0f} s")
    print(f"  VERIFIABLE      : {ver_ok}/{len(ver)}  "
          f"({100 * ver_ok / max(len(ver), 1):.0f} %)"
          "   <- la bonne valeur, prouvee")
    print(f"  FORME           : {forme_ok}/{len(forme)}  "
          f"({100 * forme_ok / max(len(forme), 1):.0f} %)"
          "   <- bonne formation seulement")
    print("  par categorie :")
    for cat in sorted({r["categorie"] for r in resultats}):
        sous = [r for r in resultats if r["categorie"] == cat]
        n_ok = sum(1 for r in sous if r["ok"])
        lat = [r["latence"] for r in sous]
        print(f"    {cat:12s} {n_ok:3d}/{len(sous):3d}   "
              f"latence med {sorted(lat)[len(lat) // 2]:6.2f}s "
              f"max {max(lat):6.2f}s")
    # PROVENANCE (QUI a repondu ?) : le score seul ne dit pas quel etage
    # de la cascade a produit la reponse — un changement d'ORDRE peut
    # degrader le systeme sans bouger d'un point. C'est le meme besoin
    # que scripts/sous_banc.py (mesure rapide de provenance), rendu ici
    # lisible sur le banc complet.
    print("  provenance (qui a repondu) :")
    par_cerveau: dict = {}
    for r in resultats:
        par_cerveau.setdefault(r.get("cerveau") or "?", []).append(
            bool(r["ok"]))
    provenance = {}
    for nom, oks in sorted(par_cerveau.items(),
                           key=lambda kv: -len(kv[1])):
        nb = sum(oks)
        provenance[nom] = f"{nb}/{len(oks)}"
        print(f"    {nom:22s} {nb:3d}/{len(oks):3d}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"score": f"{bons}/{total}",
                        "score_verifiable": f"{ver_ok}/{len(ver)}",
                        "score_forme": f"{forme_ok}/{len(forme)}",
                        "provenance": provenance,
                        "duree_s": round(duree, 1),
                        "resultats": resultats},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.json}")
    return 0 if bons == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
