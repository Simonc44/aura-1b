"""Recalibre le detecteur de concepts sur l'historique reel des conversations.

Trois controles, tous deterministes et reproductibles :

1. DISTRIBUTION (aura/.conversation.jsonl) : sur les questions >= _MOTS_MIN
   mots, combien recoivent un LEXIQUE (>= seuil), un REPLI DE DOMAIN
   (bande d'indice) ou le SILENCE — c'est la part reelle de traffic
   concerne par la bibliotheque.

2. AUDIT DE FAUX POSITIFS : un hit dont la question ne contient AUCUN
   mot-cle du domaine du concept retenu est suspect (ex. meteo ->
   effet-Matthieu mesure au calibrage). Les suspects sont listes ; si le
   seuil peut les couper sans blesser les vrais hits, il est propose.

3. CALIBRATION SUR JEU DE REFERENCE (paires requete->concept mesurees au
   micro-eval) : le seuil courant doit toujours separer positifs et
   negatifs ; un ecart net -> proposition de valeur.

Usage :
  python scripts/recalibrer_concepts.py             # rapport (dry-run)
  python scripts/recalibrer_concepts.py --ecrire    # + brouillons manquants
                                                    #   dans concepts.toml
  python scripts/recalibrer_concepts.py --chemin X  # autre historique
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from aura import concepts, embeddings  # noqa: E402

HISTORIQUE = RACINE / "aura" / ".conversation.jsonl"
CACHE = RACINE / "aura" / ".cache_reponses.jsonl"
TOML = RACINE / "concepts.toml"

# Paires calibrees (micro-eval, cf. tests/test_concepts.py) + entrees
# extraites du savoir RAG (scripts/extraire_concepts.py) — le jeu de
# reference de verite pour le seuil. POSITIFS = (requete, concept attendu).
POSITIFS = [
    ("est ce que le libre arbitre existe si tout est determine par nos genes",
     "determinisme-liberte"),
    ("si tous les A sont B et que x est A alors x est B est ce valide",
     "syllogisme"),
    ("deux variables bougent ensemble est ce que l une cause l autre",
     "causalite-vs-correlation"),
    ("est ce que je devrais toujours maximiser le bonheur du plus grand "
     "nombre meme si ca coute une personne", "utilitarisme"),
    ("quel est le symbole chimique de l oxygene gazeux", "sciences-exactes"),
    ("quelle est la capitale de l italie aujourdhui", "geographie-monde"),
    ("qu est ce qu un navigateur internet pour aller sur le web",
     "informatique-bases"),
    ("combien de temps faut il dormir la nuit pour un adulte",
     "sante-hygiene"),
    ("difference entre un fichier et un dossier sous windows",
     "informatique-bases"),
]
NEGATIFS = [
    "calcule 2 puissance 10 s il te plait",
    "quel temps fait il a paris demain sans pluie",
]

# questions NON conceptuelles (pour l'audit) : usage quotidien courant
_RE_NON_CONCEPT = re.compile(
    r"\b(prix|heure|meteo|temps fait|blague|bonjour|salut|qui es tu|"
    r"combien (fait|coute|dure)|calcule|addition|multiplication|"
    r"quelle est la capitale|capitale de)\b")


def _normaliser(texte: str) -> str:
    brut = unicodedata.normalize("NFKD", texte.lower())
    brut = "".join(c for c in brut if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in brut).split())


def _meilleur_score(texte: str, noms, matrice) -> tuple[str, float]:
    """(concept top-1, score) meme SOUS le seuil — pour la distribution."""
    vec = embeddings.encoder1(texte)
    if vec is None or matrice is None:
        return "", 0.0
    sims = matrice @ vec
    i = int(np.argmax(sims))
    return noms[i], float(sims[i])


def _domaine_corrobore(question: str, concept: str) -> bool:
    """La question contient-elle un mot-cle du domaine de ce concept ?"""
    domaine = concepts._DOMAINES.get(concept)
    if not domaine:
        return True                    # pas de domaine = pas d'audit
    bloc, mots_cles = concepts._BLOCS_DOMAINES.get(domaine, ("", ()))
    if not bloc:
        return True
    q = _normaliser(question)
    return any(m in q for m in mots_cles)


def _questions_historique(chemin: Path) -> list[str]:
    if not chemin.is_file():
        return []
    questions = []
    for ligne in chemin.read_text(encoding="utf-8",
                                  errors="ignore").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            obj = json.loads(ligne)
        except json.JSONDecodeError:
            continue
        if obj.get("role") == "user":
            texte = (obj.get("content") or "").strip()
            if texte:
                questions.append(texte)
    # dedup en conservant l'ordre (les memes questions reviennent au banc)
    vus, uniques = set(), []
    for q in questions:
        cle = _normaliser(q)
        if cle and cle not in vus:
            vus.add(cle)
            uniques.append(q)
    return uniques


def _jeu_reference(noms, matrice) -> tuple[list[float], list[float]]:
    """Scores (positifs = score du concept ATTENDU, negatifs = top-1)."""
    positifs: list[float] = []
    negatifs: list[float] = []
    for q, attendu in POSITIFS:
        vec = embeddings.encoder1(q)
        if vec is None or matrice is None or attendu not in noms:
            continue
        sims = matrice @ vec
        positifs.append(float(sims[noms.index(attendu)]))
        top = noms[int(np.argmax(sims))]
        if top != attendu:
            print(f"  ~ top-1={top} (attendu {attendu}) : {q[:60]}")
    for q in NEGATIFS:
        _c, score = _meilleur_score(q, noms, matrice)
        negatifs.append(score)
    return positifs, negatifs


def rapport(chemin: Path) -> dict:
    noms, matrice = concepts._vecteurs_concepts() or ([], None)
    if not noms:
        print("[recalibre] bibliotheque ou encodeur indisponible")
        return {}
    seuil = concepts.seuil()

    # ── 1. distribution sur l'historique reel ──
    questions = [q for q in _questions_historique(chemin)
                 if len(q.split()) >= concepts._MOTS_MIN]
    total_brut = len(_questions_historique(chemin))
    lexique = repli = silence = 0
    suspects: list[tuple[str, str, float]] = []
    scores: list[float] = []
    manquants: list[tuple[str, str, float]] = []
    for q in questions:
        concept, score = _meilleur_score(q, noms, matrice)
        scores.append(score)
        hit = concepts.detecter(q)
        if hit and not hit.get("repli"):
            lexique += 1
            if not _domaine_corrobore(q, concept):
                suspects.append((q, concept, score))
        elif hit and hit.get("repli"):
            repli += 1
        else:
            silence += 1
            if score >= 0.25 and not _RE_NON_CONCEPT.search(q.lower()):
                manquants.append((q, concept, score))

    print(f"[recalibre] historique : {chemin.name} — {total_brut} questions, "
          f"{len(questions)} exploitables (>= {concepts._MOTS_MIN} mots)")
    if questions:
        scores_tries = sorted(scores)
        p50 = scores_tries[len(scores_tries) // 2]
        p90 = scores_tries[int(len(scores_tries) * 0.9)]
        print(f"  scores top-1 : p50={p50:.3f} p90={p90:.3f} "
              f"seuil={seuil:.2f}")
        print(f"  LEXIQUE {lexique} ({100 * lexique / len(questions):.0f}%) | "
              f"REPLI {repli} | SILENCE {silence} "
              f"({100 * silence / len(questions):.0f}%)")

    # ── 2. audit des faux positifs suspects ──
    print(f"[recalibre] hits sans corroboration de domaine : "
          f"{len(suspects)} suspect(s)")
    for q, concept, score in suspects[:8]:
        print(f"  ! {score:.3f} {concept:26s} | {q[:70]}")

    # ── 3. calibration sur le jeu de reference ──
    positifs, negatifs = _jeu_reference(noms, matrice)
    if positifs and negatifs:
        min_pos, max_neg = min(positifs), max(negatifs)
        print(f"[recalibre] reference : positifs {min_pos:.3f}.."
              f"{max(positifs):.3f} | negatifs {max(negatifs):.3f}.."
              f"{min(negatifs):.3f}")
        if max_neg < min_pos:
            propose = round((max_neg + min_pos) / 2, 2)
            statut = ("OK (le seuil courant separe toujours le jeu)"
                      if abs(propose - seuil) < 0.01
                      else f"PROPOSE {propose}")
            print(f"  ecart net -> {statut}")
        else:
            print("  ecart CHEVAUCHE -> seuil conserve (les faux positifs "
                  "ne sont pas separables du vrai par le seuil seul)")

    # ── 4. brouillons de themes manquants recurrents ──
    print(f"[recalibre] {len(manquants)} question(s) proche(s) du seuil "
          "sans concept (candidats a l'extension)")
    for q, concept, score in manquants[:10]:
        print(f"  ? {score:.3f} (proche {concept}) | {q[:70]}")
    return {"lexique": lexique, "repli": repli, "silence": silence,
            "suspects": suspects, "manquants": manquants}


def ecrire_brouillons(manquants: list[tuple[str, str, float]]) -> int:
    """Themes manquants recurrents (>= 2 questions partageant un mot
    significatif) -> brouillons ajoutes a concepts.toml (dry-run sinon)."""
    # reponses reelles du cache (q -> r) pour fonder les brouillons
    reponses: dict[str, str] = {}
    if CACHE.is_file():
        for ligne in CACHE.read_text(encoding="utf-8",
                                     errors="ignore").splitlines():
            try:
                obj = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            q, r = (obj.get("q") or "").strip(), (obj.get("r") or "").strip()
            if q and r:
                reponses[_normaliser(q)] = r
    # mot significatif partage par >= 2 questions manquantes
    groupes: dict[str, list[str]] = {}
    for q, _c, _s in manquants:
        mots = [m for m in _normaliser(q).split()
                if len(m) >= 5 and m not in ("quelle", "comment", "est-ce")]
        for m in mots:
            groupes.setdefault(m, []).append(q)
    themes = {m: qs for m, qs in groupes.items() if len(qs) >= 2}
    if not themes:
        print("[recalibre] aucun theme manquant recurrent — rien a ecrire")
        return 0
    # charge l'existant pour ne pas ecraser
    existantes: dict[str, str] = {}
    if TOML.is_file():
        import tomllib
        try:
            with open(TOML, "rb") as f:
                existantes = {str(k): str(v) for k, v in
                              (tomllib.load(f).get("concepts") or {}).items()}
        except Exception:  # noqa: BLE001
            pass
    for mot, questions in sorted(themes.items()):
        nom = f"q-{mot}"
        if nom in existantes or nom in concepts._CONCEPTS:
            continue
        morceaux = []
        for q in questions[:3]:
            rep = reponses.get(_normaliser(q))
            morceaux.append(f"{q} -> {rep}" if rep else f"{q} -> (reponse a confirmer)")
        existantes[nom] = ("questions frequentes de l'utilisateur sur "
                           + mot + " : " + " ; ".join(morceaux))
        print(f"  + brouillon {nom} ({len(questions)} question(s))")
    lignes = ["# Concepts etendus (voir scripts/extraire_concepts.py et",
              "# scripts/recalibrer_concepts.py).",
              "", "[concepts]"]
    for cle in sorted(existantes):
        lignes.append(f"{cle} = {json.dumps(existantes[cle], ensure_ascii=False)}")
    TOML.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"[recalibre] {len(existantes)} cle(s) -> {TOML.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    chemin = HISTORIQUE
    if "--chemin" in argv:
        i = argv.index("--chemin")
        chemin = Path(argv[i + 1])
    resultats = rapport(chemin)
    if "--ecrire" in argv and resultats:
        return ecrire_brouillons(resultats.get("manquants") or [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
