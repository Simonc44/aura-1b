"""Routeur intelligent : decide quels experts activer pour chaque question.

Trois methodes, du plus simple au plus fiable :
1. Mots-clés (fallback, zero dependance) — ancien comportement
2. Embeddings TF-IDF + similarité cosinus — aucun entraînement requis
3. Classifieur TF-IDF + LogisticRegression — 50+ exemples par categorie

Les prototypes d'embeddings et le classifieur sont entraînés au premier appel,
puis mis en cache sur disque (fichier .joblib). Le routeur ne bloque pas au
démarrage.
"""
import logging
import os
from pathlib import Path

import numpy as np

LOG = logging.getLogger("aura.routeur")

# -- dataset d'entrainement ------------------------------------------------
# 50+ exemples par categorie : le routeur apprend meme les synonymes et fautes

_EXEMPLES_MATH = [
    "calcule 2 puissance 10",
    "resous l equation x carré moins 5x plus 6",
    "quel est le resultat de 15 divisé par 3",
    "combien fait 7 facteurielle",
    "trouve la formule pour ces donnees numeriques",
    "découvre la loi mathematique de ce graphique",
    "quel est le carre de 12",
    "resous cette equation differentielle",
    "calcule l integrale de x carré",
    "quel est le PGCD de 48 et 36",
    "calcule le pourcentage de 25 sur 200",
    "resous le systeme d equations",
    "combien vaut la racine carree de 144",
    "quel est le resultat de 3 puissance 4",
    "découvre la relation entre x et y",
    "calcule la derivative de x cube",
    "resous 2x plus 3 egale 11",
    "quel est le minimum de cette fonction",
    "calcule la variance de ces nombres",
    "trouve le nombre manquant dans cette suite",
    "combien fait 100 modulo 7",
    "resous cette inéquation",
    "quel est le produit scalaire de ces vecteurs",
    "calcule l aire de ce cercle de rayon 5",
    "découvre la loi physique de cette experience",
    "resous cette递归 recurrence",
    "combien vaut log de 100",
    "quel est le determinant de cette matrice",
    "calcule la pente de cette droite",
    "trouve l error de ce model",
    "resous cette optimisation",
    "quel est le barycentre de ces points",
    "calcule la distance entre ces deux points",
    "découvre le pattern dans ces données",
    "resous ce probleme de combinatoire",
    "combien fait 12 choose 5",
    "quel est le zéro de cette fonction",
    "calcule la convolution de ces signaux",
    "trouve la valeur propre de cette matrice",
    "resous ce systeme lineaire",
    "quel est le rang de cette matrice",
    "calcule la transposé de cette matrice",
    "découvre la loi exponentielle de ces données",
    "resous cette équation du second degré",
    "combien vaut e puissance 2",
    "quel est le plus petit commun multiple",
    "calcule la moyenne pondérée",
    "trouve le seuil optimal de ce classifieur",
    "resous ce probleme de flot maximum",
    "quel est le coefficient de corrélation",
]

_EXEMPLES_WEB = [
    "qui a gagne la coupe du monde 2026",
    "quelle est la derniere actualite en France",
    "quel est le prix du bitcoin aujourd'hui",
    "qui est le president des etats unis",
    "quelle est la meteo a paris demain",
    "quel est le dernier film sorti au cinema",
    "qui a invente l internet",
    "quelle est la population de la chine en 2026",
    "quel est le record du monde d athletisme",
    "qui a ecris la petite maison dans la prairie",
    "quelle est la capitale de l australie",
    "quel est le meilleur restaurant de lyon",
    "qui a gagne le ballon d or 2025",
    "quelle est la dernière version de python",
    "quel est le taux de chomage en france",
    "qui est le fondateur d apple",
    "quelle est la decouverte recente en physique",
    "quel est le prix de l or aujourd'hui",
    "qui a gagne les olympiques 2024",
    "quelle est la densite de population de tokyo",
    "quel est le dernier telephone sorti",
    "qui est le plus riche du monde",
    "quelle est la distance terre lune",
    "quel est le record de vitesse du son",
    "qui a gagne le grammy 2026",
    "quelle est la temperature sur mars",
    "quel est le dernier conflit geopolitique",
    "qui est le directeur general de google",
    "quelle est la decouverte recente en biologie",
    "quel est le taux d inflation en europe",
    "qui a ecris le petit prince",
    "quelle est la capitale du bresil",
    "quel est le plus grand fleuve du monde",
    "qui a gagne le prix nobel de physique 2025",
    "quelle est la vitesse de la lumiere",
    "quel est le dernier buzz sur twitter",
    "qui est le president de la russie",
    "quelle est la composition de l air",
    "quel est le record du monde de natation",
    "qui a gagne la ligue des champions 2026",
    "quelle est la densite de l eau",
    "quel est le dernier scout en space",
    "qui est le patron de tesla",
    "quelle est la decouverte recente en intelligence artificielle",
    "quel est le taux de satisfaction des francais",
    "qui a gagne le tour de france 2026",
    "quelle est la capitale de la Coree du Sud",
    "quel est le prix du petrole",
    "qui est le plus grand realisateur de tous les temps",
    "quelle est la dernière actualite sportive",
]

_EXEMPLES_GENERAL = [
    "bonjour comment ca va",
    "raconte moi une blague",
    "quel temps fait il",
    "comment tu t appelles",
    "merci beaucoup",
    "au revoir",
    "explique moi ce qu est l intelligence artificielle",
    "donne moi un conseil pour etudier",
    "quel est le sens de la vie",
    "raconte moi une histoire",
    "comment faire du pain maison",
    "quel est ton plat prefere",
    "donne moi une citation inspirante",
    "comment bien dormir",
    "quel livre recommandes tu",
    "comment se faire des amis",
    "quel est le meilleur conseil que tu puisse donner",
    "explique moi la gravite",
    "quel est le but de l humanite",
    "donne moi une recette de cuisine",
    "comment devenir plus productif",
    "quel est le premier restaurant du monde",
    "comment gagner en confiance",
    "quel est le meilleur sport",
    "donne moi un resume de ton travail",
    "comment prendre soin de sa sante",
    "quel est ton avis sur l informatique quantique",
    "comment apprendre une nouvelle langue",
    "quel est le plus beau voyage",
    "donne moi un resume de ce que tu sais",
    "comment gerer le stress",
    "quel est le meilleur conseil pour un debutant en programmation",
    "comment bien manger",
    "quel est le role d un serveur web",
    "donne moi un exemple de code python",
    "comment fonctionne un reseau de neurones",
    "quel est le meilleur framework web",
    "comment structurer un projet informatique",
    "quel est le role d un administrateur systeme",
    "donne moi un resume de cette page web",
    "comment trouver un stage en informatique",
    "quel est le meilleur langage pour debuter",
    "comment preparer un entretien technique",
    "quel est le role d un data scientist",
    "donne moi un resume de ce concept",
    "comment ameliorer son profil github",
    "quel est le meilleur outil de gestion de projet",
    "comment ecrire un bon README",
    "quel est le role d un tech lead",
    "donne moi un conseil pour coder plus vite",
]

# -- prototypes par categorie (phrases representatives) --------------------

_PROTOTYPES = {
    "math": [
        "calcule cette formule mathematique",
        "resous cette equation",
        "decouvre la loi entre x et y",
        "quel est le resultat de ce calcul",
        "trouve la formule exacte",
    ],
    "web": [
        "quelle est la derniere actualite",
        "qui a fait cette decouverte recemment",
        "quel est le prix actuel de",
        "derniere nouvelle en science",
        "meteo temperature record",
    ],
    "general": [
        "bonjour comment vas tu",
        "raconte moi quelque chose",
        "explique moi ce concept",
        "donne moi un conseil",
        "quelle est ton opinion sur",
    ],
}

# -- classes ----------------------------------------------------------------

class RouteurIntelligent:
    """Classe la question et renvoie les experts a activer."""

    def __init__(self, dossier_cache: str | Path | None = None):
        self._dossier_cache = Path(dossier_cache or os.path.join(
            os.path.dirname(__file__), ".cache_routeur"))
        self._classifieur = None
        self._vectoriseur = None
        self._proto_vecs = None
        self._proto_labels = None

    # -- initialisation paresseuse -----------------------------------------

    def _entrainer_classifieur(self):
        """Enregistre le TF-IDF + LogReg sur le dataset de 150 exemples."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        questions = _EXEMPLES_MATH + _EXEMPLES_WEB + _EXEMPLES_GENERAL
        labels = (["math"] * len(_EXEMPLES_MATH) +
                  ["web"] * len(_EXEMPLES_WEB) +
                  ["general"] * len(_EXEMPLES_GENERAL))

        self._vectoriseur = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), max_features=2000)
        X = self._vectoriseur.fit_transform(questions)
        self._classifieur = LogisticRegression(max_iter=200, C=1.0)
        self._classifieur.fit(X, labels)

        # sauvegarder sur disque
        try:
            import joblib
            self._dossier_cache.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._vectoriseur, self._dossier_cache / "vectoriseur.joblib")
            joblib.dump(self._classifieur, self._dossier_cache / "classifieur.joblib")
            LOG.info("[routeur] classifieur entraine et sauvegarde (%d exemples)", len(questions))
        except ImportError:
            LOG.info("[routeur] classifieur entraine (pas de joblib, pas de cache)")

    def _charger_ou_entrainer(self):
        if self._classifieur is not None:
            return
        try:
            import joblib
            v = self._dossier_cache / "vectoriseur.joblib"
            c = self._dossier_cache / "classifieur.joblib"
            if v.exists() and c.exists():
                self._vectoriseur = joblib.load(v)
                self._classifieur = joblib.load(c)
                LOG.info("[routeur] classifieur charge depuis le cache")
                return
        except ImportError:
            pass
        self._entrainer_classifieur()

    def _charger_protos(self):
        """Calcule les vecteurs TF-IDF des prototypes (une seule fois)."""
        if self._proto_vecs is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vect = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=2000)
        corpus = []
        self._proto_labels = []
        for label, phrases in _PROTOTYPES.items():
            for p in phrases:
                corpus.append(p)
                self._proto_labels.append(label)
        self._proto_vecs = vect.fit_transform(corpus)
        self._proto_vect = vect

    # -- classification -----------------------------------------------------

    def classer(self, question: str) -> dict:
        """Classe une question et renvoie {'methodes', 'scores', 'experts'}.

        Experts = set de noms : 'math', 'web', 'general'
        """
        q = question.lower().strip()
        if not q:
            return {"methodes": ["vide"], "scores": {}, "experts": {"general"}}

        # methode 1 : classifieur TF-IDF + LogReg
        self._charger_ou_entrainer()
        X = self._vectoriseur.transform([q])
        pred = self._classifieur.predict(X)[0]
        probas = dict(zip(self._classifieur.classes_, self._classifieur.predict_proba(X)[0]))

        # methode 2 : similarité cosinus avec prototypes
        self._charger_protos()
        from sklearn.metrics.pairwise import cosine_similarity
        vec_q = self._proto_vect.transform([q])
        sims = cosine_similarity(vec_q, self._proto_vecs)[0]
        best_proto = self._proto_labels[np.argmax(sims)]
        score_proto = float(sims.max())

        # fusion des deux methodes
        experts = set()
        experts.add(pred)
        if score_proto > 0.15:
            experts.add(best_proto)

        # regle de securite : si les deux methodes disent "math", on renforce
        if pred == best_proto:
            experts = {pred}

        return {
            "methodes": ["LogReg", "cosinus"],
            "scores": {k: round(v, 3) for k, v in probas.items()},
            "proto_score": round(score_proto, 3),
            "proto_label": best_proto,
            "prediction": pred,
            "experts": experts,
        }

    def routeur_classique(self, question: str) -> dict:
        """Fallback : mots-clés simples (aucune dependance ML)."""
        q = question.lower()
        mots_math = ("calcule", "resous", "equation", "puissance", "carre",
                     "racine", "formule", "données", "donnees", "loi",
                     "relation", "x²", "y=", "integrale", "derivee")
        mots_web = ("qui est", "qui a", "actualite", "dernier", "prix",
                     "meteo", "record", "capitale", "2026", "2025",
                     "decouverte", "découverte", "invente")
        e = set()
        if any(m in q for m in mots_math):
            e.add("math")
        if any(m in q for m in mots_web):
            e.add("web")
        if not e:
            e.add("general")
        return {"methodes": ["mots_cles"], "scores": {}, "experts": e}
