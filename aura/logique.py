"""Résolveur logique exact par backtracking (pur Python, zéro dépendance).

Même coup que PoT (aura/raisonneur.py) mais pour la logique SANS nombres :
le 1B FORMALISE le puzzle en contraintes, le programme DÉDUIT exactement
par énumération systématique. Domaines cibles : chevaliers/menteurs,
attributions (qui fait quoi / qui a quel age / quelle couleur).

Formalisme attendu du modele (tolerant a la casse et aux espaces) :

    ENTITES : Paul, Marie, Leo
    DOMAINE : chevalier, menteur
    TOUS DIFFERENTS
    CONDITION : Marie == menteur
    Paul DIT Marie == menteur

Conventions :
- le PREMIER domaine = la valeur « dit vrai » (pour DIT : le locuteur
  chevalier dit vrai, le menteur ment) ;
- expressions : ==, !=, ET, OU, NON, parenthèses ; a droite de ==, un nom
  d'ENTITE compare les valeurs (Paul == Marie) sinon une valeur du domaine ;
- le solveur refuse tout ce qu'il ne comprend pas (jamais de devinette).

Resultat : toutes les solutions. UNIQUE -> reponse exacte ; MULTIPLE ou
AUCUNE -> on refuse et le pipeline retombe sur le chemin normal.
"""
import itertools
import logging
import re
import unicodedata

LOG = logging.getLogger("aura.logique")

# -- expressions --------------------------------------------------------------
# grammaire : OU(ET(NON(ATOM)))  — priorité croissante, parenthèses autorisées

_TOK = re.compile(r"\s*(\(|\)|==|!=|ET\b|OU\b|NON\b|[A-Za-zÀ-ÿ_][\w'À-ÿ]*)")


def _tokeniser(expr: str) -> list[str]:
    tokens, pos = [], 0
    while pos < len(expr):
        m = _TOK.match(expr, pos)
        if not m:
            if expr[pos].isspace():
                pos += 1
                continue
            raise ValueError(f"symbole inattendu : {expr[pos]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


class _Parser:
    """Recursive descent : expr -> OU -> ET -> NON -> ATOM."""

    def __init__(self, tokens: list[str], entites: set[str]):
        self.toks = tokens
        self.i = 0
        self.entites = entites

    def _voir(self) -> str:
        return self.toks[self.i] if self.i < len(self.toks) else ""

    def _manger(self) -> str:
        t = self._voir()
        self.i += 1
        return t

    def parser(self):
        arbre = self._ou()
        if self.i != len(self.toks):
            raise ValueError(f"tokens en surplus : {self.toks[self.i:]}")
        return arbre

    def _ou(self):
        gauche = self._et()
        while self._voir() == "OU":
            self._manger()
            droite = self._et()
            gauche = ("ou", gauche, droite)
        return gauche

    def _et(self):
        gauche = self._non()
        while self._voir() == "ET":
            self._manger()
            droite = self._non()
            gauche = ("et", gauche, droite)
        return gauche

    def _non(self):
        if self._voir() == "NON":
            self._manger()
            return ("non", self._non())
        return self._atome()

    def _atome(self):
        if self._voir() == "(":
            self._manger()
            arbre = self._ou()
            if self._manger() != ")":
                raise ValueError("parenthese fermante attendue")
            return arbre
        gauche = self._manger()
        if gauche not in self.entites:
            raise ValueError(f"entite inconnue : {gauche!r}")
        op = self._manger()
        if op not in ("==", "!="):
            raise ValueError(f"operateur attendu, lu : {op!r}")
        droite = self._manger()
        if not droite or droite in ("ET", "OU", "NON", "(", ")"):
            raise ValueError(f"comparande manquant pour {gauche!r}")
        if droite in self.entites:
            return (op, ("var", gauche), ("var", droite))
        return (op, ("var", gauche), ("val", droite))


def _evaluer(arbre, env: dict, valeurs) -> bool:
    op = arbre[0]
    if op == "non":
        return not _evaluer(arbre[1], env, valeurs)
    if op == "et":
        return _evaluer(arbre[1], env, valeurs) and _evaluer(arbre[2], env, valeurs)
    if op == "ou":
        return _evaluer(arbre[1], env, valeurs) or _evaluer(arbre[2], env, valeurs)
    (kind_g, g), (kind_d, d) = arbre[1], arbre[2]
    vg = env[g] if kind_g == "var" else g
    vd = env[d] if kind_d == "var" else d
    return (vg == vd) if op == "==" else (vg != vd)


# -- parsing du puzzle ---------------------------------------------------------

_L_ENTITES = re.compile(r"^ENTITES\s*:\s*(.+)$", re.IGNORECASE)
_L_DOMAINE = re.compile(r"^DOMAINE\s*:\s*(.+)$", re.IGNORECASE)
_L_DIT = re.compile(r"^(\w[\w'À-ÿ]*)\s+DIT\s+(.+)$", re.IGNORECASE)
_L_CONDITION = re.compile(r"^(?:CONDITION|SI)\s*:\s*(.+)$", re.IGNORECASE)
_L_DIFF = re.compile(r"^TOUS\s+DIFFERENTS$", re.IGNORECASE)
_L_REPONSE = re.compile(r"^(REPONSE|SOLUTION)\s*:", re.IGNORECASE)


def parser_puzzle(texte: str) -> dict:
    """Parse le formalisme (tolerant : prefixe 'CONDITION :' optionnel).

    Leve ValueError a la moindre ligne illicite : jamais de devinette.
    """
    entites: list[str] = []
    domaine: list[str] = []
    contraintes: list[tuple] = []
    tous_diff = False
    for brut in texte.splitlines():
        ligne = unicodedata.normalize("NFKD", brut.strip())
        ligne = "".join(c for c in ligne if not unicodedata.combining(c))
        # le modele colle parfois une glose entre parentheses a une ligne
        # valide (« TOUS DIFFERENTS (seulement si...) ») : on la retire
        ligne = re.sub(r"\s*\([^()]*\)\s*$", "", ligne).strip()
        if not ligne or ligne.startswith("#") or _L_REPONSE.match(ligne):
            continue
        if _L_DIFF.match(ligne):
            tous_diff = True
            continue
        m = _L_ENTITES.match(ligne)
        if m:
            entites = [e.strip() for e in m.group(1).split(",") if e.strip()]
            continue
        m = _L_DOMAINE.match(ligne)
        if m:
            domaine = [v.strip() for v in m.group(1).split(",") if v.strip()]
            continue
        m = _L_DIT.match(ligne)
        if m:
            contraintes.append(("dit", m.group(1), m.group(2)))
            continue
        m = _L_CONDITION.match(ligne)
        if m:
            contraintes.append(("cond", m.group(1)))
            continue
        # condition sans prefixe (le modele les saute parfois)
        if "==" in ligne or "!=" in ligne:
            contraintes.append(("cond", ligne))
            continue
        raise ValueError(f"ligne non comprise : {brut.strip()!r}")

    if len(entites) < 2:
        raise ValueError("il faut au moins 2 entites")
    if len(set(e.lower() for e in entites)) != len(entites):
        raise ValueError("entites dupliquees")
    ents = set(entites)
    for c in contraintes:
        if c[0] == "dit" and c[1] not in ents:
            raise ValueError(f"locuteur inconnu : {c[1]!r}")

    arbres: list[tuple] = []
    for c in contraintes:
        if c[0] == "cond":
            arbres.append(("cond", _Parser(_tokeniser(c[1]), ents).parser()))
        else:
            arbres.append(("dit", c[1],
                           _Parser(_tokeniser(c[2]), ents).parser()))

    # le modele melange parfois placeholders (v1...) et vraies valeurs :
    # si AUCUNE valeur declaree n'apparait dans les contraintes, on infere
    # le domaine des valeurs observees (ordre d'apparition conserve)
    vals: list[str] = []

    def _collecter(a):
        if a[0] in ("et", "ou"):
            _collecter(a[1])
            _collecter(a[2])
        elif a[0] == "non":
            _collecter(a[1])
        else:
            (op, (kg, g), (kd, d)) = a
            if kd == "val" and d not in vals:
                vals.append(d)

    for c in arbres:
        _collecter(c[1] if c[0] == "cond" else c[2])
    if vals and domaine and not any(v in domaine for v in vals) \
            and 2 <= len(vals) <= 8:
        LOG.info("[logique] domaine infere des contraintes : %s", vals)
        domaine = vals

    if not (2 <= len(domaine) <= 8):
        raise ValueError("domaine : 2 a 8 valeurs attendues")
    if tous_diff and len(domaine) < len(entites):
        raise ValueError("TOUS DIFFERENTS exige un domaine >= entites")
    return {"entites": entites, "domaine": domaine,
            "contraintes": arbres, "tous_diff": tous_diff}


def resoudre(puzzle: dict, max_solutions: int = 3) -> dict:
    """Enumere les solutions (backtracking naive : <= 8^6 cas, instantane)."""
    entites = puzzle["entites"]
    domaine = puzzle["domaine"]
    vrai = domaine[0]
    solutions = []

    def satisfait(env) -> bool:
        for c in puzzle["contraintes"]:
            if c[0] == "cond":
                if not _evaluer(c[1], env, domaine):
                    return False
            else:                       # S DIT expr : verite == assertion
                dit_vrai = env[c[1]] == vrai
                if dit_vrai != _evaluer(c[2], env, domaine):
                    return False
        return True

    for combinaison in itertools.product(domaine, repeat=len(entites)):
        if puzzle["tous_diff"] and len(set(combinaison)) != len(entites):
            continue
        env = dict(zip(entites, combinaison))
        if satisfait(env):
            solutions.append(env)
            if len(solutions) >= max_solutions:
                break
    return {"solutions": solutions,
            "statut": ("unique" if len(solutions) == 1
                       else "multiple" if solutions else "aucune")}


def bloc_verification(puzzle: dict, solutions: list[dict]) -> str:
    """Bloc injecte au modele : LA solution deduite exactement."""
    lignes = ["SOLUTION DEDUITE EXACTEMENT PAR LE RESOLVEUR LOGIQUE :"]
    for ent in puzzle["entites"]:
        lignes.append(f"- {ent} : {solutions[0][ent]}")
    lignes.append("Cette solution satisfait TOUTES les contraintes du puzzle.")
    return "\n".join(lignes)


_SYSTEME_LOGIQUE = (
    "Tu transformes l'enigme DONNEE en formalisme. Reponds UNIQUEMENT avec "
    "ces lignes (aucune prose, aucun titre invente) :\n"
    "ENTITES : <les noms EXACTS de l'enigme, separes par des virgules>\n"
    "DOMAINE : <les valeurs EXACTES possibles, separees par des virgules>\n"
    "TOUS DIFFERENTS          (seulement si chaque entite a une valeur distincte)\n"
    "CONDITION : <Nom> == <valeur>        (une contrainte par ligne, != possible)\n"
    "<Nom> DIT <Nom> == <valeur>          (pour les chevaliers/menteurs)\n"
    "\n"
    "Regles absolues :\n"
    "- recopie les noms et valeurs TELS QUELS depuis l'enigme\n"
    "- N'INVENTE aucune contrainte qui n'est pas dans l'enigme\n"
    "- chaque ligne commence par son mot-cle (ENTITES, DOMAINE, CONDITION...)"
)
