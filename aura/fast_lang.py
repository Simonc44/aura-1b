"""FastLang : le dictionnaire symbolique O(1) — la langue sans hallucination.

Là où le 1B devine (« le arbre », conjugaison approximative, définition
inventée), ce moteur CONSULTE : une base SQLite embarquée dans le paquet
(aura/lang.db, ouverte en lecture seule), réponses en < 0,2 ms, zéro LLM.
Thread-safe : une connexion par thread (sqlite3 n'est pas partageable
entre threads), le fichier n'est jamais écrit.

Trois tables (construites par scripts/construire_lang.py) :
  conjugaisons(verbe, temps, personne) -> forme   (ex. etre, futur, je -> serai)
  dictionnaire(mot) -> definition                 (définitions exactes)
  elisions(mot) -> elision                        (exceptions : h aspiré)

Principe : tout ce qui peut être une TABLE ne doit jamais être GÉNÉRÉ.
La base voyage dans le .aef (la forge empaquette aura/).
"""
from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
from pathlib import Path

_BDD = Path(__file__).parent / "lang.db"

# voyelle initiale (élision autorisée même sans la base)
_RE_VOYELLE = re.compile(r"^[aeiouyéèêàâîôûùœ]")
# définition demandée : « qu'est-ce que X », « définis X », « que veut dire X »
_RE_DEF_MOT = re.compile(
    r"(?:qu'?est.?ce.?que(?:\s+la|\s+le|\s+l')?|definis|définis|"
    r"que\s+veut\s+dire|definition\s+de|définition\s+de)\s+"
    r"([a-zà-ÿœ][\w-]+)", re.IGNORECASE)

_TEMPS = ("present", "imparfait", "futur", "passe_compose")


def _sans_accents(texte: str) -> str:
    n = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in n if not unicodedata.combining(c))


class FastLangEngine:
    """Consultation symbolique : conjugaison, définition, élision.

    Une connexion SQLite par thread (thread-local), en lecture seule
    (URI mode=ro) : le fan-out web ∥ PGS et l'orchestrateur peuvent
    consulter en même temps, sans verrou, sans jamais écrire la base.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._chemin = Path(db_path) if db_path else _BDD
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection | None:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if not self._chemin.exists():
                return None
            conn = sqlite3.connect(f"file:{self._chemin}?mode=ro", uri=True,
                                   check_same_thread=False)
            self._local.conn = conn
        return conn

    def _verbe_exact(self, verbe: str) -> str | None:
        """Le vrai verbe de la base pour une saisie avec/sans accents
        (« etre » -> « être »). None si inconnu."""
        conn = self._conn()
        if conn is None:
            return None
        verbe = verbe.lower().strip()
        if conn.execute("SELECT 1 FROM conjugaisons WHERE verbe=? LIMIT 1",
                        (verbe,)).fetchone():
            return verbe
        for (v,) in conn.execute("SELECT DISTINCT verbe FROM conjugaisons"):
            if _sans_accents(v) == _sans_accents(verbe):
                return v
        return None

    # -- conjugaison ---------------------------------------------------------

    def get_conjugation(self, verbe: str, temps: str, personne: str) -> str | None:
        """Forme conjuguée exacte (< 0,2 ms), None si verbe/temps inconnu.

        temps : present | imparfait | futur | passe_compose
        personne : je | tu | il (il/elle) | nous | vous | ils (ils/elles)
        La forme stockée est SANS pronom (« serai », « ai été ») :
        composable, non ambiguë.
        """
        conn = self._conn()
        if conn is None:
            return None
        t = _sans_accents(temps).replace(" ", "_")
        if t not in _TEMPS:
            return None
        verbe_exact = self._verbe_exact(verbe)
        if verbe_exact is None:
            return None
        row = conn.execute(
            "SELECT forme FROM conjugaisons WHERE verbe=? AND temps=? "
            "AND personne=? LIMIT 1",
            (verbe_exact, t, personne.lower())).fetchone()
        return row[0] if row else None

    def conjuguer_table(self, verbe: str, temps: str) -> dict[str, str] | None:
        """Les 6 personnes d'un temps : {'je': 'serai', ...} ou None."""
        conn = self._conn()
        if conn is None:
            return None
        t = _sans_accents(temps).replace(" ", "_")
        if t not in _TEMPS:
            return None
        verbe_exact = self._verbe_exact(verbe)
        if verbe_exact is None:
            return None
        lignes = conn.execute(
            "SELECT personne, forme FROM conjugaisons "
            "WHERE verbe=? AND temps=?", (verbe_exact, t)).fetchall()
        return dict(lignes) if len(lignes) == 6 else None

    # -- dictionnaire ----------------------------------------------------------

    def get_definition(self, mot: str) -> str | None:
        """Définition exacte, sans hallucination. None si mot inconnu —
        et None = cascade normale (ancrage web), jamais une invention.
        Tolérant aux accents : « photosynthese » (sans accent dans la
        question) trouve l'entrée « photosynthèse »."""
        conn = self._conn()
        if conn is None:
            return None
        mot = mot.lower().strip()
        row = conn.execute(
            "SELECT definition FROM dictionnaire WHERE mot=? LIMIT 1",
            (mot,)).fetchone()
        if row:
            return row[0]
        for m, definition in conn.execute(
                "SELECT mot, definition FROM dictionnaire"):
            if _sans_accents(m) == _sans_accents(mot):
                return definition
        return None

    # -- élision -----------------------------------------------------------------

    def elision(self, determinant: str, suivant: str) -> str | None:
        """Verdict d'élision pour « <determinant> <suivant> ».

        Renvoie :
          « l' »  — élision obligatoire (« le arbre » -> « l'arbre »)
          « le »/« la » — h ASPIRÉ, pas d'élision (« le héros »)
          None    — cas non concerné (autre déterminant, consonne)
        La base porte les exceptions connues ; à défaut, règle générale :
        voyelle -> élision, h minuscule -> élision (h muet par défaut),
        h majuscule (nom propre : Hollande) -> pas d'élision.
        """
        det = (determinant or "").lower()
        mot = (suivant or "").strip()
        if det not in ("le", "la") or not mot:
            return None
        bas = mot.lower()
        conn = self._conn()
        if conn is not None:
            row = conn.execute(
                "SELECT elision FROM elisions WHERE mot=? LIMIT 1",
                (bas,)).fetchone()
            if row:
                return row[0]
        if _RE_VOYELLE.match(bas):
            return "l'"
        if bas.startswith("h") and not mot[0].isupper():
            return "l'"          # h muet par défaut (hérisson, honte...)
        return None

    def disponible(self) -> bool:
        return self._chemin.exists()

    # -- détecteur orchestrateur ----------------------------------------------

    def repondre(self, question: str) -> str | None:
        """Réponse déterministe si la question est une demande de LANGUE
        connue de la base. None = cascade normale (jamais d'invention).

        Reconnu : « conjugue <verbe> (au présent/au futur/à l'imparfait/
        au passé composé) » et « qu'est-ce que X / définis X » quand X
        a une définition exacte dans la base.
        """
        if not question:
            return None
        q = _sans_accents(question)
        if "conjugue" in q or "conjugaison" in q:
            conn = self._conn()
            if conn is None:
                return None
            for (verbe,) in conn.execute(
                    "SELECT DISTINCT verbe FROM conjugaisons ORDER BY verbe"):
                if re.search(rf"\b{re.escape(_sans_accents(verbe))}\b", q):
                    temps = "present"
                    if "futur" in q:
                        temps = "futur"
                    elif "imparfait" in q:
                        temps = "imparfait"
                    elif "passe" in q:
                        temps = "passe_compose"
                    table = self.conjuguer_table(verbe, temps)
                    if not table:
                        return None
                    etiq = temps.replace("_", " ")
                    return (f"Au {etiq}, le verbe « {verbe} » se conjugue : "
                            f"je {table['je']}, tu {table['tu']}, "
                            f"il/elle {table['il']}, nous {table['nous']}, "
                            f"vous {table['vous']}, "
                            f"ils/elles {table['ils']}.")
            return None
        m = _RE_DEF_MOT.search(question)
        if m:
            mot = m.group(1).lower()
            definition = self.get_definition(mot)
            if definition:
                return f"{mot} : {definition}"
        return None


# instance partagée (les connexions restent thread-local en interne)
moteur = FastLangEngine()
