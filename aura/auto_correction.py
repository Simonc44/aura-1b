"""Self-Correction Stream : verifier la PENSEE du modele pendant qu'il pense.

Le probleme
-----------
Un modele qui raisonne en <think> ecrit des calculs INTERMEDIAIRES avant de
conclure. Sur un 1B, ces calculs sont la principale source d'erreur : le
raisonnement part d'un « 17 x 3 = 51 » et la reponse finale, pourtant bien
redigee, est fausse. Le raisonnement n'est PAS une preuve.

Ce que fait ce module
--------------------
Il lit la pensee AU FIL DE L'EAU et y cherche ce qui est VERIFIABLE
EXACTEMENT : les egalites arithmetiques (`17 x 3 = 51`). Chaque membre de
gauche est evalue sur des rationnels exacts — jamais des flottants, une
erreur d'arrondi ne doit pas produire une fausse accusation. Une egalite
fausse devient une ALERTE, avec la valeur juste.

Que faire de l'alerte
---------------------
Il faut distinguer ce qui est possible de ce qui ne l'est pas :

* INSERER un token d'alerte DANS le flux en cours est impossible ici. Les
  deux chemins d'inference sont un generateur Python (in-process) et un
  flux SSE HTTP (llama-server) : ni l'un ni l'autre n'accepte de token en
  retour pendant la generation.
* ARRETER et RELANCER est possible, et produit le meme effet
  (reevaluation), au prix d'une generation. C'est donc ce qui est fait :
  une seule relance, uniquement si une erreur EXACTE a ete trouvee, avec la
  correction injectee dans le prompt. L'echeance du tour est respectee
  (une relance qui n'y tient pas n'est pas lancee).

Le cout est borne : au plus UNE relance par tour, et seulement sur une
contradiction demontrable. Un modele qui pense juste ne paie rien.

Ce que ce module n'est pas
--------------------------
Ce n'est pas un juge de la verite. Il ne sait verifier que des egalites
arithmetiques exactes, et il le dit : `verifier()` renvoie None quand il n'a
rien de solide a dire. Un controleur qui pretend verifier ce qu'il ne peut
pas est pire que pas de controleur du tout.
"""
from __future__ import annotations

import os
import re
from fractions import Fraction

#: Les quatre operations + la puissance. `x`/`X` et `×` sont des
#: multiplications (le modele ecrit les deux) ; `**` et `^` des puissances.
_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}

#: Nombre : entier, decimal, ou separateur de milliers (1 024 / 1,024).
_NB = r"\d[\d\u00a0 ,]*\.?\d*"
_RE_EGALITE = re.compile(
    rf"(?<![\d.])(?P<g>{_NB})\s*(?P<op>\*\*|[+\-*/x×^])\s*(?P<d>{_NB})"
    rf"\s*(?:=|≈|\\approx)\s*(?P<r>{_NB})"
)


def _nombre(brut: str) -> Fraction:
    """'1 024' / '1,024' / '2.5' -> Fraction exacte."""
    net = brut.replace("\u00a0", "").replace(" ", "").replace(",", "")
    return Fraction(net)


def _calcule(gauche: str, op: str, droite: str) -> Fraction | None:
    """Valeur EXACTE du membre de gauche, ou None si non calculable.

    None (plutot qu'une exception) est le contrat : l'appelant ne doit
    jamais voir planter un controleur qui lit du texte libre.
    """
    try:
        a, b = _nombre(gauche), _nombre(droite)
    except (ValueError, ZeroDivisionError):
        return None
    if op in ("**", "^"):
        # puissance : entiere seulement, et bornee (2^100000 n'est pas un
        # controle, c'est un deni de service sur la memoire).
        if b.denominator != 1 or not (0 <= b <= 4096):
            return None
        return a ** int(b)
    if op == "/" and b == 0:
        return None
    return _OPS["*" if op in ("x", "×") else op](a, b)


def verifier(texte: str) -> str | None:
    """La premiere egalite FAUSSE de `texte`, sinon None.

    Renvoie une note courte et actionnable — pas une exception, pas un
    verdict global : c'est une observation locale, datee dans la pensee.
    """
    for m in _RE_EGALITE.finditer(texte or ""):
        juste = _calcule(m.group("g"), m.group("op"), m.group("d"))
        if juste is None:
            continue
        try:
            affirme = _nombre(m.group("r"))
        except (ValueError, ZeroDivisionError):
            continue
        if juste != affirme:
            op = m.group("op").replace("x", "×").replace("*", "×")
            return (f"{m.group('g').strip()} {op} {m.group('d').strip()} "
                    f"ne font pas {m.group('r').strip()} mais {_joli(juste)}")
    return None


def _joli(valeur: Fraction) -> str:
    """Fraction -> texte lisible (1024 plutot que 1024/1)."""
    if valeur.denominator == 1:
        return str(valeur.numerator)
    return f"{float(valeur):.6g}"


def note_pour_prompt(alerte: str) -> str:
    """La note injectee dans le prompt de la relance.

    Adressee au modele, pas a l'humain : elle dit ce qui est faux et
    demande explicitement de REVERIFIER cette etape, sans dicter la fin de
    la reponse (le modele doit corriger le raisonnement, pas subir un
    resultat).
    """
    return ("VERIFICATION EXACTE DE TON RAISONNEMENT : tu as ecrit "
            f"« {alerte} ». Le calcul a ete refait exactement et il est "
            "faux. Reprends ce point, corrige-le, puis donne ta reponse "
            "finale.")


def actif() -> bool:
    """Interrupteur (AURA_AUTOCORRECTION=0 coupe). Actif par defaut."""
    return os.environ.get("AURA_AUTOCORRECTION", "1") != "0"


class SurveillancePensee:
    """Suit la pensee d'un tour et retient LA premiere alerte exacte.

    Alimentee par le repartiteur de flux (aura/llama_cerveau._FluxTour), qui
    est le seul endroit qui voit la reflexion AVANT la reponse. On analyse
    par PHRASE, pas a chaque fragment : une egalite coupee en deux
    (« 17 x 3 = » puis « 51 ») ne doit pas etre jugee sur sa moitie.
    """

    #: Longueur maximale gardee d'un coup dans le tampon d'analyse : au-dela,
    #: la pensee est deja loin, et le tampon ne doit pas croitre sans fin.
    _MAX_TAMPON = 4096

    def __init__(self, actif_: bool | None = None) -> None:
        self.actif = actif() if actif_ is None else actif_
        self.alerte: str | None = None
        self._tampon = ""

    def pousser(self, morceau: str) -> None:
        """Un fragment de PENSEE (jamais de reponse finale)."""
        if not self.actif or self.alerte is not None:
            return
        self._tampon += morceau
        # on ne juge que les phrases TERMINEES
        morceaux = re.split(r"(?<=[.!?;\n])", self._tampon)
        self._tampon = morceaux.pop() if morceaux else ""
        if len(self._tampon) > self._MAX_TAMPON:
            self._tampon = self._tampon[-self._MAX_TAMPON:]
        for phrase in morceaux:
            alerte = verifier(phrase)
            if alerte is not None:
                self.alerte = alerte
                return

    def terminer(self) -> str | None:
        """Fin de generation : juge la derniere phrase, puis renvoie l'alerte."""
        if self.actif and self.alerte is None and self._tampon.strip():
            self.alerte = verifier(self._tampon)
        return self.alerte

    def reinitialiser(self) -> None:
        """Repart a zero sur la PENSEE SUIVANTE (apres une relance).

        Sans ca, l'alerte deja trouvee resterait posee et couperait aussi la
        generation corrigee : la correction se saboterait elle-meme.
        """
        self.alerte = None
        self._tampon = ""
