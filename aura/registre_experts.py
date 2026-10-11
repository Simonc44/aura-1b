"""Registre des experts : l'ORDRE de la cascade est une DONNEE, pas du code.

Probleme mesure (6 octobre 2026)
--------------------------------
L'ordre de consultation vivait dans l'ordre des `if` de
`orchestrateur.executer_detaille`. Consequences constatees :

  * le cache semantique (approximatif, seuil 0,70) passait AVANT
    `calcul_taux` (exact) — il a servi 240 a une question de taux ;
  * deux questions du banc changeaient de resultat selon cet ordre ;
  * ajouter un expert etait un pari : ou le placer ?
  * personne ne pouvait relire cet ordre, ni le tester.

Ici chaque expert est DECLARE avec son rang, son genre et sa raison. Le
genre porte l'invariant verifiable : **aucun etage approximatif ne peut etre
declare avant un etage exact**. `verifier()` le controle, et
`tests/test_registre_experts.py` verifie en plus que la cascade consulte
reellement les experts dans cet ordre (un expert oublie dans le branchement
fait echouer le test, au lieu de disparaitre en silence).

Ajouter un expert = ajouter une entree ICI + un appel et une garde dans
orchestrateur._appel_expert / _expert_applicable. Les trois sont relies par
le test : le registre est la source de verite, pas un commentaire.

Meme principe pour une PORTE (cf. PORTES plus bas) : une etape qui ne
repond pas mais qui peut ARRETER la cascade (aujourd'hui la clarification
interactive). Sa position dans la cascade est declaree ici, pas deduite de
l'endroit ou l'appel a ete ecrit.
"""
from __future__ import annotations

from dataclasses import dataclass

# Genres, du plus fiable au plus approximatif. Le rang porte l'invariant.
EXACT = "exact"           # calcul prouve — aucune hallucination possible
VERIFIE = "verifie"       # savoir indexe et relu (rag-binaire)
FLOU = "flou"             # memoire approximative a seuil (cache semantique)
GENERATIF = "generatif"   # necessite le LLM

RANG_GENRE: dict[str, int] = {EXACT: 0, VERIFIE: 1, FLOU: 2, GENERATIF: 3}


@dataclass(frozen=True)
class Expert:
    nom: str        # etiquette publique == cerveau_choisi
    genre: str
    cout: str       # ordre de grandeur, pas une mesure
    source: str     # ou vit l'implementation
    note: str = ""  # POURQUOI ce rang (la raison survit au refactor)

    @property
    def rang(self) -> int:
        return RANG_GENRE[self.genre]


@dataclass(frozen=True)
class Contexte:
    """Ce dont les gardes ont besoin — passe par la boucle, pas de global."""
    question: str
    est_puzzle: bool
    a_donnees: bool          # le pipeline a recu (X, y) : calcul numerique


# ── L'ORDRE DECLARE ───────────────────────────────────────────────────────
# Toute entree doit avoir son appel + sa garde dans l'orchestrateur : le
# test de registre echoue sinon (c'est le garde-fou contre l'oubli).
EXPERTS: tuple[Expert, ...] = (
    Expert("fast-lang", EXACT, "< 0,2 ms", "fast_lang.moteur.repondre",
           "Conjugaison / definition connue de la base SQLite locale. "
           "ATTENTION : le banc 100 ne pose AUCUNE de ces questions (0/100) "
           "— ce 0 mesure la couverture du BANC, pas l'utilite du tier "
           "(il repond juste et instantanement a « conjugue parler au "
           "present »). Sondes dediees dans scripts/sous_banc.py."),
    Expert("calcul-achat", EXACT, "< 1 ms", "calcul_achat.resoudre",
           "AVANT calcul-verbal, obligatoirement : « j'achete » y est lu "
           "comme une ADDITION (« j'ai 15 euros et j'achete un livre a "
           "7 euros » rendait 22 au lieu de 8)."),
    Expert("calcul-verbal", EXACT, "< 1 ms", "calcul_verbal.resoudre",
           "Ajout / retrait / moitie racontes, traduits par regles."),
    Expert("calcul-taux", EXACT, "< 1 ms", "calcul_taux.resoudre",
           "APRES verbal, volontairement : mesure — verbal rend None sur "
           "les 5 questions de taux du banc, donc ce tier n'ajoute QUE du "
           "rappel (il ne peut rien retirer a verbal)."),
    Expert("chaines-numeriques", EXACT, "< 1 ms", "chaines_numeriques.resoudre",
           "Ancrage + ecarts chiffres (« Alice a 3 ans de plus que Bob, "
           "Clara a 4 ans, quel age a Alice ? ») et extremum sur valeurs "
           "explicites. Ajoute le 6 oct. 2026 : 4 questions du banc "
           "passaient au 1B alors qu'elles sont de l'algebre elementaire. "
           "AVANT experts-speciaux : pur calcul, aucun appel au modele."),
    Expert("problemes-arithmetiques", EXACT, "< 1 ms",
           "problemes_arithmetiques.resoudre",
           "Fractions en mots (« aux deux tiers de 60 ») et figures simples "
           "(perimetre / aire d'un rectangle) : deux etapes, une seule "
           "quantite en entree, donc invisibles pour les solveurs a une "
           "etape (filtre_instantane, calcul-verbal, calcul-taux). "
           "Ajoute le 7 oct. 2026 apres mesure du sous-banc : 3 questions "
           "de calcul pur tombaient encore sur le LLM (~40 s chacune). "
           "AVANT experts-speciaux : pur calcul, aucun appel au modele."),
    Expert("micro-faits", EXACT, "< 1 ms", "micro_faits.resoudre",
           "Savoir elementaire chiffre (durees, cotes des polygones, couleurs "
           "primaires, photosynthese). Ajoute le 8 oct. 2026 apres mesure du "
           "sous-banc : 4 questions que le 1B rate par manque de RAPPEL, pas "
           "de raisonnement, tombaient sur le modele (factuel-08/11/12 et "
           "format-couleurs). Table COURTE et abstention stricte : le savoir "
           "etendu reste a rag-binaire. AVANT experts-speciaux parce que "
           "celui-ci, seul « exact » au cout generatif, peut appeler le 1B."),
    Expert("experts-speciaux", EXACT, "5 ms a > 1 s",
           "orchestrateur._expert_special",
           "tri-transitif, SAT (logique), sandbox (potcode). En dernier des "
           "exacts car la formalisation SAT et potcode appellent le 1B : "
           "c'est le seul « exact » dont le cout est generatif."),
    Expert("rag-binaire", VERIFIE, "~5 ms", "rag_binaire.chercher",
           "Index local de faits relus. Exclu sur un puzzle : un puzzle se "
           "DEDUIT (PoT + AST), il ne se documente pas — sinon l'index sert "
           "une enigme a la place du solveur (regression vue au banc)."),
    Expert("cache-semantique", FLOU, "~1 ms",
           "filtre_instantane.repondre_cache",
           "Memoire approximative (cosinus MiniLM >= 0,70 sur 384 "
           "dimensions). APRES tous les exacts et apres le savoir verifie — "
           "c'etait exactement le bug corrige le 6 oct. 2026. Reste AVANT "
           "toute generation : une question deja vue ne coute toujours rien."),
)

# ── LES PORTES ────────────────────────────────────────────────────────────
# Une porte ne PROPOSE pas de reponse : elle peut ARRETER la cascade pour
# demander une precision (aura/clarification.py). Elle n'a donc pas de rang
# de fiabilite. En revanche sa POSITION est un invariant, et c'est celui qui
# compte : posee AVANT la derniere couche qui repond, elle refuserait de
# traiter une question que le systeme sait deja trancher — exactement le
# defaut du cache semantique (corrige le 6 oct. 2026), a l'envers.
APRES_EXPERTS = "apres-experts"

POSITIONS: dict[str, str] = {
    APRES_EXPERTS: "apres la derniere couche qui repond, avant la generation",
}


@dataclass(frozen=True)
class Porte:
    nom: str        # etiquette publique == cerveau_choisi
    position: str
    source: str     # ou vit l'implementation
    note: str = ""  # POURQUOI cette position (la raison survit au refactor)


PORTES: tuple[Porte, ...] = (
    Porte("clarification", APRES_EXPERTS, "clarification.detecter",
          "Question trop ouverte (« Explique », « Ecris un poeme » sans "
          "sujet, « Convertis 100 » sans unite cible) : plutot que de laisser "
          "le 1B inventer un sujet, la porte pose 1 a 3 questions et ne "
          "repond pas. APRES les experts, parce qu'une question exacte doit "
          "rester exacte ; AVANT la generation, parce que demander une "
          "precision coute moins cher qu'une reponse fausse. Un seul tour : "
          "la precision recollee relance le pipeline sans nouvelle porte."),
)


# Gardes : nom -> condition d'applicabilite. Declare ici pour que le test
# puisse exiger une garde pour CHAQUE expert du registre.
GARDES: dict[str, str] = {
    "rag-binaire": "not est_puzzle",
    "cache-semantique": "not a_donnees",
}

# NB : "fast-lang" et "tri-transitif" ne sont pas des etages SEPARES ici.
# - fast-lang est bien un expert a part entiere (premier de la liste) ;
# - tri-transitif vit a l'interieur d'`experts-speciaux` (avec le SAT et
#   potcode) : le decouper changerait l'etiquette publique `tri-transitif`
#   que scripts/banc_100.py et le sous-banc suivent deja.


def noms() -> tuple[str, ...]:
    """Les experts, dans l'ORDRE DECLARE (c'est cet ordre que la cascade suit)."""
    return tuple(e.nom for e in EXPERTS)


def expert(nom: str) -> Expert:
    for e in EXPERTS:
        if e.nom == nom:
            return e
    raise KeyError(nom)


def porte(nom: str) -> Porte:
    for p in PORTES:
        if p.nom == nom:
            return p
    raise KeyError(nom)


def verifier_portes() -> list[str]:
    """Violations des invariants de PORTE. Liste vide = portes saines."""
    pbs: list[str] = []
    noms_portes = [p.nom for p in PORTES]
    if len(set(noms_portes)) != len(noms_portes):
        pbs.append("noms de portes dupliques : %s" % (tuple(noms_portes),))
    for p in PORTES:
        if p.position not in POSITIONS:
            pbs.append("%s : position inconnue %r" % (p.nom, p.position))
        if p.nom in noms():
            pbs.append("%s est a la fois expert et porte : deux contrats "
                       "differentiels pour le meme nom" % p.nom)
    return pbs


def verifier() -> list[str]:
    """Violations de l'invariant d'ordre. Liste vide = registre sain.

    Un etage approximatif declare AVANT un etage plus fiable est une
    violation : c'est le defaut qui a fait passer le cache avant les
    solveurs exacts. Les portes sont controlees par le meme point d'entree
    pour qu'un ajout ne puisse pas echapper au garde-fou.
    """
    pbs: list[str] = verifier_portes()
    if len(set(noms())) != len(EXPERTS):
        pbs.append("noms d'experts dupliques : %s" % (noms(),))
    for i, e in enumerate(EXPERTS):
        if e.genre not in RANG_GENRE:
            pbs.append("%s : genre inconnu %r" % (e.nom, e.genre))
        for avant in EXPERTS[:i]:
            if avant.rang > e.rang:
                pbs.append(
                    "%s (%s) est declare avant %s (%s) — un etage moins "
                    "fiable passe devant un etage plus fiable"
                    % (e.nom, e.genre, avant.nom, avant.genre))
    return pbs
