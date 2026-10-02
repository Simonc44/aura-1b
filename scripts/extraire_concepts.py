"""Extrait des concepts.toml depuis le savoir RAG verifie.

La bibliotheque de concepts (aura/concepts.py) etait figee a 30 entrees
ecrites a la main. Ce script la nourrit des MEMES sources verifies que
scripts/construire_rag.py :
  - datasets/{web,general,code}.jsonl : paires (question, reponse)
  - aura/.graphe_faits.jsonl          : triplets verifies (s, r, o)
  - knowledge.jsonl / aura/knowledge.jsonl : savoir manuel (si present)

Methode (deterministe, reviewable) :
  1. les questions sont nettoyees des preambules (« Dis-moi : », ...) ;
  2. chaque entree est grouppee par THeme (mot-cles identifies dans la
     question) — un theme devient un candidat de concept ;
  3. un theme avec >= 2 faits verifies produit une description COURTE
     (2-3 faits « question -> reponse »), la calibration montrant que les
     descriptions longues degradent la detection ;
  4. NOUVEAUTE : un candidat trop proche d'un concept deja present
     (cosinus >= 0.80 sur les descriptions) est ecarte — pas de doublon.

Usage :
  python scripts/extraire_concepts.py            # rapport (dry-run)
  python scripts/extraire_concepts.py --ecrire   # ecrit/complete concepts.toml
  python scripts/extraire_concepts.py --forcer   # + ecrase les cles existantes
"""
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from aura import concepts, embeddings  # noqa: E402

SORTIE = RACINE / "concepts.toml"

# Preambules de politesse a retirer des questions (recurrents datasets)
_PREAMBULES = (
    "peux-tu m'expliquer", "peux tu m expliquer", "explique-moi", "explique moi",
    "j'aimerais savoir", "j aimerais savoir", "dis-moi", "dis moi",
    "donne-moi le code pour", "donne moi le code pour", "donne-moi",
    "donne moi", "quelle est", "quel est", "qu'est-ce que", "quest ce que",
    "c'est quoi", "c est quoi",
)

# theme -> (libelle injecte, mots-cles sur la question NORMALISEE).
# L'ordre compte : une entree rejoint le PREMIER theme qui matche.
_THEMES: dict[str, tuple[str, tuple[str, ...]]] = {
    "cuisine-alimentation": (
        "cuisine et alimentation",
        ("cuisson", "cuire", "pate", "pates", "recette", "aliment", "four",
         "pain", "boisson", "ingredient", "assaisonn", "saisir", "griller",
         "dejeuner", "petit dejeuner", "repas")),
    "sante-hygiene": (
        "sante et hygiene de vie",
        ("sante", "maladie", "microbe", "medicament", "douleur", "laver",
         "mains", "sommeil", "dormir", "cancer", "risque", "symptome",
         "hygiene", "tabac", "bacterie", "virus")),
    "geographie-monde": (
        "geographie du monde",
        ("capitale", "pays", "drapeau", "continent", "nation", "ocen",
         "montagne", "fleuve", "ville la plus")),
    "langue-francaise": (
        "langue francaise : sens et differences",
        ("difference entre", "signifie", "orthographe", "traduire",
         "traduction", "conjugaison", "grammaire", "se dit", "synonyme",
         "ecrire un mot")),
    "sciences-exactes": (
        "sciences : chimie et physique",
        ("symbole chimique", "molecule", "element chimique", "reaction",
         "atome", "acide", "metal", "physique", "energie", "force de",
         "vitesse de la lumiere")),
    "informatique-bases": (
        "informatique : bases et environnement",
        ("ordinateur", "fichier", "dossier", "navigateur", "logiciel",
         "clavier", "ecran", "windows", "internet", "reseau", "capture d",
         "menu", "parametre")),
    "programmation": (
        "programmation et code",
        ("python", "fonction", "boucle", "variable", "algorithme", "script",
         "programme", "liste", "chaines de caracteres", "convertit",
         "developper")),
    "culture-generale": (
        "culture generale : objets et usages du quotidien",
        ("qu'est-ce que c", "a quoi sert", "comment fonctionne",
         "testament", "symbole", "pronom", "outil", "appareil")),
}


def _normaliser(texte: str) -> str:
    import unicodedata
    brut = unicodedata.normalize("NFKD", texte.lower())
    brut = "".join(c for c in brut if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in brut).split())


def _nettoyer_question(question: str) -> str:
    q = " ".join(question.split())
    modifie = True
    while modifie:
        modifie = False
        bas = q.lower().strip(" ?!.,:")
        for p in _PREAMBULES:
            if bas.startswith(p):
                q = q[len(p):].lstrip(" \t:?,.!?")
                modifie = True
                break
    return q.strip() or question.strip()


def _lire_sources() -> list[dict]:
    entrees: list[dict] = []
    # datasets (math exclu : domaine exact niveau 0/PGS, cf. construire_rag)
    dossier = RACINE / "datasets"
    for chemin in sorted(dossier.glob("*.jsonl")) if dossier.exists() else []:
        if chemin.stem == "math":
            continue
        for ligne in chemin.read_text(encoding="utf-8",
                                      errors="ignore").splitlines():
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                obj = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            q = (obj.get("question") or "").strip()
            r = (obj.get("reponse") or obj.get("answer") or "").strip()
            if q and r and len(q) >= 6:
                entrees.append({"q": _nettoyer_question(q), "a": r,
                                "src": chemin.stem})
    # graphe de faits verifies : triplets -> paire (question, valeur)
    graphe = RACINE / "aura" / ".graphe_faits.jsonl"
    if graphe.exists():
        for ligne in graphe.read_text(encoding="utf-8",
                                      errors="ignore").splitlines():
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                t = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            s, rel, o = (t.get("s") or "").strip(), (t.get("r") or "").strip(), \
                (t.get("o") or "").strip()
            if s and o:
                entrees.append({"q": f"quel est le {rel} {s}".strip(),
                                "a": o, "src": "graphe"})
    # savoir manuel
    for nom in ("knowledge.jsonl", "aura/knowledge.jsonl"):
        chemin = RACINE / nom
        if not chemin.exists():
            continue
        for ligne in chemin.read_text(encoding="utf-8",
                                      errors="ignore").splitlines():
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                obj = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            q = (obj.get("question") or obj.get("q")
                 or obj.get("titre") or "").strip()
            r = (obj.get("reponse") or obj.get("a") or obj.get("texte")
                 or obj.get("contenu") or "").strip()
            if q and r:
                entrees.append({"q": _nettoyer_question(q),
                                "a": r[:200], "src": "knowledge"})
    return entrees


def _grouper(entrees: list[dict]) -> dict[str, list[dict]]:
    groupes: dict[str, list[dict]] = {t: [] for t in _THEMES}
    vus: set[tuple[str, str]] = set()
    for e in entrees:
        q_norm = _normaliser(e["q"])
        for theme, (_libelle, mots_cles) in _THEMES.items():
            if any(m in q_norm for m in mots_cles):
                cle = (theme, q_norm)
                if cle not in vus:
                    vus.add(cle)
                    groupes[theme].append(e)
                break
    return {t: e for t, e in groupes.items() if len(e) >= 2}


def _description(theme: str, faits: list[dict], max_cle: int = 340) -> str:
    libelle = _THEMES[theme][0]
    morceaux = []
    longueur = len(libelle) + 3
    for f in faits:
        q = " ".join(f["q"].split())[:80]
        a = " ".join(f["a"].split())[:110]
        morceau = f"{q} -> {a}"
        if longueur + len(morceau) + 3 > max_cle:
            break
        morceaux.append(morceau)
        longueur += len(morceau) + 3
    return f"{libelle} : " + " ; ".join(morceaux)


def _trop_proche(description: str, vec, biblio: list[tuple[str, str]]) -> bool:
    """Doublon sémantique d'un concept deja present (cosinus >= 0.80)."""
    if vec is None:
        return False
    noms = [d for _n, d in biblio]
    matrice = embeddings.encoder(noms)
    if matrice is None:
        return False
    sims = matrice @ vec
    return bool(len(sims) and float(max(sims)) >= 0.80)


def extraire(ecrire: bool = False, forcer: bool = False) -> int:
    entrees = _lire_sources()
    groupes = _grouper(entrees)
    print(f"[concepts] {len(entrees)} entrees lues -> "
          f"{len(groupes)} themes retenus (>= 2 faits)")

    # bibliotheque existante (integree + concepts.toml deja la)
    deja = dict(concepts._CONCEPTS)
    existantes = list(deja.items())

    candidats: dict[str, str] = {}
    rejetes: list[tuple[str, str]] = []
    for theme, faits in groupes.items():
        nom = theme
        description = _description(theme, faits)
        if nom in deja and not forcer:
            rejetes.append((nom, "cle deja presente"))
            continue
        vec = embeddings.encoder1(description)
        if _trop_proche(description, vec, existantes):
            rejetes.append((nom, "trop proche d'un concept existant"))
            continue
        candidats[nom] = description
        existantes.append((nom, description))
        print(f"  + {nom:24s} ({len(faits)} faits, {len(description)} signes)")
    for nom, motif in rejetes:
        print(f"  - {nom:24s} ({motif})")

    if not candidats:
        print("[concepts] aucun nouveau concept — concepts.toml inchange")
        return 0
    if not ecrire:
        print(f"[concepts] dry-run : {len(candidats)} candidat(s), "
              "relancer avec --ecrire pour ecrire concepts.toml")
        return 0

    # merge : on conserve les cles deja dans le fichier (sauf --forcer)
    existant_fichier: dict[str, str] = {}
    if SORTIE.is_file():
        try:
            import tomllib
            with open(SORTIE, "rb") as f:
                bloc = (tomllib.load(f).get("concepts") or {})
            existant_fichier = {str(k): str(v) for k, v in bloc.items()
                                if isinstance(v, str)}
        except Exception as e:  # noqa: BLE001
            print(f"[concepts] {SORTIE.name} illisible ({e}) -> recreation")
    for cle, valeur in candidats.items():
        if forcer or cle not in existant_fichier:
            existant_fichier[cle] = valeur

    lignes = [
        "# Concepts etendus — genere par scripts/extraire_concepts.py",
        "# (savoir RAG verifie : datasets/, graphe de faits, knowledge).",
        "# Complete/surcharge les 30 concepts integres de aura/concepts.py ;",
        "# detection : AURA_CONCEPTS_SEUIL, desactivation : AURA_CONCEPTS=0.",
        "",
        "[concepts]",
    ]
    for cle in sorted(existant_fichier):
        lignes.append(f"{cle} = {json.dumps(existant_fichier[cle], ensure_ascii=False)}")
    SORTIE.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"[concepts] {len(existant_fichier)} cle(s) -> {SORTIE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(extraire(ecrire="--ecrire" in sys.argv,
                              forcer="--forcer" in sys.argv))
