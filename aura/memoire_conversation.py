"""Memoire de conversation : RLM (MIT), fenetre glissante, etat evenementiel.

Trois mecanismes complémentaires pour tenir un 1B dans sa fenetre :

1. RLM (Recursive Language Models, MIT) : l'historique complet vit dans
   un FICHIER externe, jamais injecte en entier. L'IA y accede par un
   OUTIL symbolique `rechercher_dans_historique(mot_cle)` — la fenetre
   du prompt reste minuscule.

2. Fenetre glissante avec ANCRAGE : les ancres fixes (message systeme,
   consignes, dissertation TAS) ne bougent jamais ; en bas, seuls les
   4 derniers tours bruts survivent. Au-dela : effet « lost in the
   middle » — un 1B n'exploite plus les vieux messages, ils ne font
   que polluer sa fenetre.

3. Etat pilote par EVENEMENTS (event-driven state) : un script pur
   extrait l'etat civil de la conversation (nom, preferences, faits
   numeriques de l'utilisateur) dans un mini-dictionnaire JSON de
   quelques octets, injecte dans le systeme. Plus besoin des phrases
   « je m'appelle Pierre » dans l'historique.

Usage orchestrateur :
    memoire = MemoireConversation()
    memoire.ajouter("user", q)
    memoire.ajouter("assistant", r)
    tours = memoire.fenetre()          # 4 derniers tours bruts
    etat  = memoire.etat_json()        # {"utilisateur": "Pierre", ...}
    hit   = memoire.rechercher("Pierre")  # outil RLM (recherche fichier)
"""
import json
import re
import threading
import time
from pathlib import Path

# ── 1. RLM : historique externalise en fichier ────────────────────────

_TAMPON = 4096                # lecture arriere par blocs de 4 Ko
_MAX_RESULTATS = 3            # resultats max renvoyes a l'IA
_MAX_EXTRAIT = 280            # taille max d'un extrait
_TAILLE_ROTATION = 32 * 1024 * 1024   # rotation : archive au-dela de 32 Mo
_MAX_ARCHIVES = 3                     # archives d'historique conservees


def _sans_accents(texte: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in n if not unicodedata.combining(c))


class MemoireConversation:
    """Historique externalise (RLM) + fenetre glissante + etat evenementiel.

    Thread-safe : l'orchestrateur est mono-thread par question, mais le
    fan-out web || PGS tourne en parallele et le RLSS/selfplay peuvent
    partager une instance — un verrou coute 0 et evite les surprises.
    """

    def __init__(self, chemin: str | Path | None = None,
                 fenetre: int = 4):
        self._verrou = threading.Lock()
        self._chemin = (Path(chemin) if chemin
                        else Path(__file__).parent / ".conversation.jsonl")
        self._fenetre = max(2, fenetre)
        # etat civil : dictionnaire plat {cle: valeur} (JSON, quelques octets)
        self._etat: dict[str, str] = {}
        # tours bruts de la session courante (pour la fenetre)
        self._tours: list[dict[str, str]] = []
        # outil RLM declare pour le prompt systeme
        self.outil_disponible = True

    # -- RLM : le fichier externe ------------------------------------------

    def _ecrire_tour(self, role: str, contenu: str) -> None:
        ligne = json.dumps({"role": role, "content": contenu},
                           ensure_ascii=False)
        with self._chemin.open("a", encoding="utf-8") as f:
            f.write(ligne + "\n")

    def _archives(self) -> list[Path]:
        """Archives de rotation de l'historique, les plus recentes d'abord."""
        try:
            return sorted(self._chemin.parent.glob(
                f"{self._chemin.stem}-*{self._chemin.suffix}"), reverse=True)
        except OSError:
            return []

    def _rotation_si_necessaire(self) -> None:
        """Archive l'historique quand le fichier RLM depasse la taille max.

        L'historique complet est le support de la memoire longue (RLM) :
        il grandit sans limite avec l'usage. Au-dela de _TAILLE_ROTATION,
        le fichier est bascule dans une archive datee
        (.conversation-AAAA-MM-JJ-HHMMSS.jsonl) et la recherche scrute le
        fichier courant PUIS les archives les plus recentes — l'ancien
        reste accessible, le fichier actif redevient petit. Les
        _MAX_ARCHIVES archives les plus recentes sont conservees.
        """
        try:
            taille = self._chemin.stat().st_size
        except OSError:
            return
        if taille < _TAILLE_ROTATION:
            return
        horodatage = time.strftime("%Y-%m-%d-%H%M%S")
        archive = self._chemin.with_name(
            f"{self._chemin.stem}-{horodatage}{self._chemin.suffix}")
        try:
            self._chemin.replace(archive)
        except OSError:
            return
        for vieille in self._archives()[_MAX_ARCHIVES:]:
            try:
                vieille.unlink()
            except OSError:
                pass

    def rechercher(self, mot_cle: str) -> str:
        """Outil RLM : recherche arriere dans le FICHIER d'historique.

        Lecture par blocs depuis la fin (le fichier et ses archives
        peuvent etre gros sans jamais etre charges entierement) ; le
        fichier courant est scrute d'abord, puis les archives de
        rotation les plus recentes. Renvoie les extraits (role +
        contexte) ou une chaine vide.
        """
        if not mot_cle or not mot_cle.strip():
            return ""
        aiguille = _sans_accents(mot_cle.strip())
        resultats: list[str] = []
        # fichier courant d'abord (memoire la plus fraiche), puis les
        # archives de rotation, les plus recentes d'abord
        chemins: list[Path] = [self._chemin, *self._archives()]
        try:
            for chemin in chemins:
                if len(resultats) >= _MAX_RESULTATS:
                    break
                if not chemin.exists():
                    continue              # fichier absent (apres rotation)
                taille = chemin.stat().st_size
                with chemin.open("rb") as f:
                    position = taille
                    while position > 0 and len(resultats) < _MAX_RESULTATS:
                        debut = max(0, position - _TAMPON)
                        f.seek(debut)
                        bloc = f.read(position - debut)
                        position = debut
                        for ligne in reversed(bloc.decode("utf-8",
                                                          errors="ignore").splitlines()):
                            if len(resultats) >= _MAX_RESULTATS:
                                break
                            ligne = ligne.strip()
                            if not ligne:
                                continue
                            try:
                                tour = json.loads(ligne)
                            except json.JSONDecodeError:
                                continue    # ligne tronquee (bord de bloc)
                            contenu = str(tour.get("content", ""))
                            if aiguille in _sans_accents(contenu):
                                resultats.append(
                                    f"[{tour.get('role', '?')}] {contenu[:_MAX_EXTRAIT]}")
        except OSError:
            return ""
        return "\n---\n".join(resultats)

    # -- fenetre glissante ---------------------------------------------------

    def ajouter(self, role: str, contenu: str) -> None:
        """Enregistre un tour : rotation eventuelle, fichier RLM, fenetre,
        extraction d'etat. La rotation passe AVANT l'ecriture : le tour
        frais demarre le nouveau fichier, l'archive garde l'ancien."""
        contenu = (contenu or "").strip()
        if not contenu:
            return
        with self._verrou:
            self._rotation_si_necessaire()
            self._ecrire_tour(role, contenu)
            self._tours.append({"role": role, "content": contenu})
            if len(self._tours) > self._fenetre * 2:   # (paire user/assistant)
                self._tours = self._tours[-self._fenetre * 2:]
            if role == "user":
                self._extraire_evenements(contenu)

    def fenetre(self) -> list[dict[str, str]]:
        """Les N derniers tours bruts (ancres en haut, fenetre en bas)."""
        with self._verrou:
            return list(self._tours)

    # -- 3. etat pilote par evenements --------------------------------------

    # declarations sur soi : « je m'appelle X », « je m appelle X », « je suis X »
    # (la normalisation sans accents conserve l'apostrophe ; le texte oral
    # sans apostrophe doit matcher aussi)
    _RE_NOM = re.compile(
        r"\b(?:je\s+m\s*'?\s*appelle|moi\s+c\s*'?\s*est|je\s+suis)\s+"
        r"([A-Z][\w-]{1,20})", re.IGNORECASE)
    # possessions chiffrees : « j'ai achete 4 serveurs »
    _RE_POSS = re.compile(
        r"\bj\s*'?\s*ai (?:achete[sr]?|acquis|recu|ajoute[sr]?|commande)\s+"
        r"(\d+)\s+([a-z\w-]{2,20})", re.IGNORECASE)
    _RE_POSS2 = re.compile(
        r"\bj\s*'?\s*ai\s+(\d+)\s+([a-z\w-]{2,20})", re.IGNORECASE)

    def _extraire_evenements(self, texte: str) -> None:
        """Extraction symbolique (regex) de l'etat civil — aucun LLM.

        Les regex tournent sur le texte normalise (sans accents), mais
        les extraits sont repris dans le texte ORIGINAL (span identique :
        la normalisation NFKD ne change pas les longueurs) pour garder
        la casse reelle (« Pierre », pas « pierre »).
        """
        original = texte.strip()
        t = _sans_accents(original)
        m = self._RE_NOM.search(t)
        if m and m.group(1).lower() not in (
                "un", "une", "la", "le", "content", "desole", "sure",
                "au courant"):
            self._etat["utilisateur"] = original[m.start(1):m.end(1)]
        m = self._RE_POSS.search(t) or self._RE_POSS2.search(t)
        if m:
            n, objet = m.group(1), m.group(2).lower()
            # eviter "j'ai 3 pommes et..." pris comme etat durable :
            # on garde tout, mais l'etat reste ecrasable (dernier etat gagne)
            if objet not in ("pas", "plus", "peu", "trop", "besoin"):
                self._etat[objet] = n

    def etat_json(self) -> str:
        """L'etat civil compact (JSON une ligne), '' si vide."""
        with self._verrou:
            if not self._etat:
                return ""
            return json.dumps(self._etat, ensure_ascii=False, sort_keys=True)

    def etat(self) -> dict[str, str]:
        with self._verrou:
            return dict(self._etat)

    # -- integration ancres + fenetre ----------------------------------------

    def bloc_systeme(self, systeme: str | None) -> str | None:
        """Injecte l'etat + l'annonce de l'outil RLM dans le message systeme.

        Les ANCRAGES (consignes, dissertation TAS) restent en haut ;
        l'etat evenementiel est ajoute en bas, compact.
        """
        morceaux = []
        if systeme:
            morceaux.append(systeme)
        etat = self.etat_json()
        if etat:
            morceaux.append(f"ETAT UTILISATEUR (fiable, a jour) : {etat}")
        if self.outil_disponible:
            morceaux.append(
                "Pour retrouver ce qui a ete dit plus tot dans la "
                "conversation, demande : « cherche dans l'historique : "
                "<mot> » — l'historique complet est consultable, il n'est "
                "pas dans ta memoire immediate.")
        return "\n".join(morceaux) if morceaux else None

    def vider(self) -> None:
        """Nouvelle conversation : fenetre + etat remis a zero (le fichier
        RLM est preserve : la memoire longue reste accessible par outil)."""
        with self._verrou:
            self._tours.clear()
            self._etat.clear()

    def tourner(self, question: str) -> str | None:
        """Applique l'outil RLM si la question demande l'historique.

        Motif : « cherche dans l'historique : X » (ou l'IA demande
        elle-meme l'acces). Renvoie le resultat de recherche a injecter,
        None sinon. C'est l'appel symbolique : zero token de prompt,
        zero LLM.
        """
        m = re.search(r"cherche dans l.historique\s*:?\s*(.+)",
                      question, re.IGNORECASE)
        if not m:
            return None
        return self.rechercher(m.group(1).strip()) or "(aucun resultat)"
