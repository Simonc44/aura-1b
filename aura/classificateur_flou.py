"""Classificateur flou : SVM lineaire (LinearSVC) sur vecteurs MiniLM.

Le routeur historique classe sur des n-grams de caracteres (TF-IDF) :
il reagit aux FORMULES. Celui-ci classe sur le SENS (vecteurs 384-d) :
il reagit aux PARAPHRASES et a la structure abstraite de la phrase
(ex. « 85 % analyse conceptuelle ouverte, pas une question factuelle »).

Les deux probas sont fondues 50/50 dans RouteurIntelligent.classer :
confiance mieux calibree -> aiguillage experts / mix LoRA / garde-fous.

- entrainement au 1er appel (encodage du savoir + fit, ~1 s, cache
  joblib ensuite — regenere si le savoir self-play a grandi) ;
- AURA_FLOU=0 coupe : le routeur garde strictement son comportement
  historique (TF-IDF seul) ;
- MiniLM absent -> probas() renvoie None, idem.
"""
import hashlib
import logging
import os
from pathlib import Path

from . import embeddings
from .routeur import RouteurIntelligent, _probas, exemples_entrainement

LOG = logging.getLogger("aura.classificateur_flou")

# meme dossier que le cache du routeur (gitignore : aura/.cache_routeur/)
_DOSSIER = Path(__file__).resolve().parent / ".cache_routeur"
_FICHIER = _DOSSIER / "classif_flou.joblib"

_classifieur = None
_cle: str | None = None     # empreinte du savoir actuellement appris
_indispo = False            # echec (pas de MiniLM/sklearn) : on n'insiste pas


def actif() -> bool:
    """AURA_FLOU != 0 et encodeur MiniLM disponible."""
    return os.environ.get("AURA_FLOU", "1") != "0" and embeddings.actif()


def _empreinte(questions: list[str]) -> str:
    return hashlib.sha1("\n".join(questions).encode("utf-8")).hexdigest()[:16]


def _entrainer() -> bool:
    """Fit LinearSVC sur les vecteurs MiniLM du savoir partage du routeur."""
    global _classifieur, _cle, _indispo
    from sklearn.svm import LinearSVC

    questions, labels = exemples_entrainement()
    X = embeddings.encoder(questions)
    if X is None:
        _indispo = True
        return False
    clf = LinearSVC(C=1.0, dual="auto", random_state=0)
    clf.fit(X, labels)
    _classifieur = clf
    _cle = _empreinte(questions)
    try:
        import joblib
        _DOSSIER.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modele": clf, "cle": _cle, "n": len(questions)},
                    _FICHIER)
    except Exception:  # noqa: BLE001  # cache best-effort
        LOG.debug("[flou] cache joblib non ecrit")
    LOG.info("[flou] LinearSVC entraine sur vecteurs MiniLM (%d exemples)",
             len(questions))
    return True


def _charger() -> bool:
    """Modele en memoire (cache disque si le savoir n'a pas bouge)."""
    global _classifieur, _cle, _indispo
    if _classifieur is not None:
        return True
    if _indispo or not actif():
        return False
    questions, _labels = exemples_entrainement()
    empreinte = _empreinte(questions)
    try:
        import joblib
        if _FICHIER.exists():
            donnees = joblib.load(_FICHIER)
            if donnees.get("cle") == empreinte:
                _classifieur = donnees["modele"]
                _cle = empreinte
                LOG.info("[flou] LinearSVC charge depuis le cache")
                return True
    except Exception:  # noqa: BLE001
        pass
    return _entrainer()


def probas(question: str) -> dict[str, float] | None:
    """Probabilites {math, web, general} sur vecteurs MiniLM, ou None.

    None = coupe (AURA_FLOU=0), modele indisponible, ou echec d encodage :
    l appelant (routeur) reste sur son avis TF-IDF historique.
    """
    if not (question or "").strip() or not actif():
        return None
    if not _charger():
        return None
    vec = embeddings.encoder1(question)
    if vec is None:
        return None
    try:
        brutes = _probas(_classifieur, vec[None, :])
        return {str(k): float(v) for k, v in brutes.items()}
    except Exception as e:  # noqa: BLE001
        LOG.info("[flou] score impossible : %s", e)
        return None
