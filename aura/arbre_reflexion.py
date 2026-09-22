"""Arbre de reflexion type : la base MCTS du Dynamic Compute.

Aujourd'hui la reflexion masquee est une CHAINE brute (<thinking>...</thinking>
filtrae par regex). Des que la recherche devient un vrai ARBRE (plusieurs
branches de reflexion evaluees puis la meilleure elargie), suivre tout cela
en chaines de caracteres devient illisible et fragile.

Ce module introduit les structures de donnees typees (dataclasses) :
- EtapeReflexion      : un pas de pensee (contenu, score, expert validateur)
- BrancheReflexion    : une hypothese complete = une liste d'etapes + score
                        global + flag 'elargie' (expansion MCTS)
- ArbreReflexion      : la recherche elle-meme : branches, selection de la
                        meilleure, tracé compact injectable dans le prompt

Les scores viennent des EXPERTS DETERMINISTES (logique.py, potcode.py,
calcul mental) — jamais du modele qui s'auto-note : c'est la meme regle
que le self-play et le RLSS. Les branches sans validateur (redaction)
recoivent un score None et sont triees apres les branches prouvees.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EtapeReflexion:
    """Un pas de pensee dans une branche (equivalent d'un noeud MCTS)."""

    contenu: str                          # le texte du pas (« 17 x 20 = 340 »)
    expert: str | None = None             # validateur : 'logique', 'potcode'…
    score: float | None = None            # note du validateur (None = non evalue)
    valide: bool | None = None            # verdict exact (None = pas de juge)

    def libelle(self) -> str:
        marqueur = {True: "[OK]", False: "[KO]", None: "[?]"}[self.valide]
        qui = f" ({self.expert})" if self.expert else ""
        return f"{marqueur}{qui} {self.contenu.strip()}"


@dataclass
class BrancheReflexion:
    """Une hypothese de resolution = un chemin dans l'arbre."""

    etapes: list[EtapeReflexion] = field(default_factory=list)
    score: float | None = None            # agregat des scores d'etapes
    elargie: bool = False                 # deja developpee (expansion MCTS)

    def ajouter(self, contenu: str, expert: str | None = None,
                score: float | None = None,
                valide: bool | None = None) -> EtapeReflexion:
        etape = EtapeReflexion(contenu, expert=expert, score=score,
                               valide=valide)
        self.etapes.append(etape)
        self._recalculer()
        return etape

    def _recalculer(self) -> None:
        """Score de branche = moyenne des etapes evaluees (None si aucune).

        Une seule etape INVALIDE ruine la branche (un raisonnement faux
        n'est pas « moyennement bon », il est faux) : score plafonne a 0.
        """
        notees = [e.score for e in self.etapes if e.score is not None]
        if any(e.valide is False for e in self.etapes):
            self.score = 0.0
        elif notees:
            self.score = sum(notees) / len(notees)
        else:
            self.score = None

    def brute(self) -> str:
        """Le texte brut de la reflexion (pour <thinking> ou le log)."""
        return "\n".join(e.contenu for e in self.etapes)

    def trace(self) -> str:
        """Trace compact avec verdicts (pour le log et le debug)."""
        lignes = [e.libelle() for e in self.etapes]
        note = f"{self.score:.2f}" if self.score is not None else "?"
        lignes.append(f"  -> score branche : {note}"
                      + (" (elargie)" if self.elargie else ""))
        return "\n".join(lignes)

    @property
    def serie(self) -> bool:
        """Branche eligible a la reponse finale : aucune etape invalide."""
        return all(e.valide is not False for e in self.etapes)


@dataclass
class ArbreReflexion:
    """La recherche MCTS-lite : branches candidates, la meilleure gagne."""

    question: str = ""
    branches: list[BrancheReflexion] = field(default_factory=list)

    def nouvelle_branche(self) -> BrancheReflexion:
        b = BrancheReflexion()
        self.branches.append(b)
        return b

    def meilleure(self) -> BrancheReflexion | None:
        """La meilleure branche : prouvees d'abord, ensuite par score.

        Les branches serie mais non evaluees (redaction) passent apres
        les branches prouvees — l'exactitude avant le style.
        """
        if not self.branches:
            return None
        triees = sorted(
            self.branches,
            key=lambda b: (b.serie, b.score if b.score is not None else -1.0),
            reverse=True)
        return triees[0]

    def trace(self) -> str:
        """Trace complet de l'arbre (log, debug, future visualisation)."""
        if not self.branches:
            return f"(arbre vide) question : {self.question}"
        blocs = [f"=== arbre : {self.question[:80]}"]
        for i, b in enumerate(self.branches, 1):
            blocs.append(f"-- branche {i}")
            blocs.append(b.trace())
        return "\n".join(blocs)


def branche_vers_prompt(branche: BrancheReflexion) -> str:
    """Injecte la branche gagnante comme brouillon deja verifie.

    La passe de redaction ne repart pas de zero : elle suit le chemin
    PROUVE (chaque etape [OK] a ete validee par un expert deterministe),
    elle ne fait que le mettre en francais.
    """
    lignes = []
    for e in branche.etapes:
        if e.valide is False:
            continue                     # les etapes rejetees ne sortent pas
        lignes.append(e.contenu.strip())
    return "\n".join(lignes)
