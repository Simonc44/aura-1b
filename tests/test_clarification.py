"""Porte de clarification : quand une question ne permet pas de repondre.

Trois familles de verrous, contre trois modes de defaillance :

1. DETECTION — les cas « trop ouverts » sont signales, les cas complets ne
   le sont PAS. Le point dur n'est pas de trouver des declencheurs, c'est de
   ne jamais declarer vide une question qui a un sujet.
2. NON-REGRESSION SUR LE BANC — les 100 questions de `scripts/banc_100.py`
   ne declenchent AUCUNE relance. Ce sont des questions reelles et
   completes : si la porte en intercepte une, le banc entier devient
   incomparable (une relance n'est pas une reponse).
3. ORDRE — la porte vit APRES la derniere couche qui repond (declaration
   `registre_experts.PORTES`) et ne peut pas se reposer une seconde fois
   (un seul tour). Les deux sont testes par le COMPORTEMENT, pas par
   relecture du code.
"""
import sys
from pathlib import Path

import pytest

_RACINE = Path(__file__).resolve().parent.parent
if str(_RACINE / "scripts") not in sys.path:
    sys.path.insert(0, str(_RACINE / "scripts"))

from aura import clarification as C  # noqa: E402
from aura import llama_cerveau


# ── 1. la detection ---------------------------------------------------------

# (question, a_contexte, famille attendue — None = question complete)
_CAS: list[tuple[str, bool, str | None]] = [
    # verbe nu : aucun objet, la reponse serait inventee de bout en bout
    ("Explique", False, "verbe-nu"),
    ("explique moi", False, "verbe-nu"),
    ("Resume", False, "verbe-nu"),
    ("Aide moi", False, "verbe-nu"),
    ("m explique", False, "verbe-nu"),              # elision sans apostrophe
    ("peux-tu m aider s il te plait", False, "verbe-nu"),
    ("aide moi a", False, "verbe-nu"),              # preposition suspendue
    ("Traduis", False, "verbe-nu"),
    ("Corrige", False, "verbe-nu"),
    ("Analyse", False, "verbe-nu"),
    ("Compare", False, "verbe-nu"),
    # objet generique : l'objet est la, il ne dit pas de quoi on parle
    ("Ecris un poeme", False, "objet-generique"),
    ("fais un plan", False, "objet-generique"),
    ("donne moi un exemple", False, "objet-generique"),
    ("Donne la liste", False, "objet-generique"),
    # renvoi sans sujet : le sens est dans un tour precedent qui n'existe pas
    ("Explique ca", False, "renvoi-sans-sujet"),
    ("Et ensuite ?", False, "renvoi-sans-sujet"),
    ("pourquoi ?", False, "renvoi-sans-sujet"),
    ("c est quoi", False, "renvoi-sans-sujet"),
    ("Continue", False, "renvoi-sans-sujet"),
    ("developpe", False, "verbe-nu"),
    # parametre manquant
    ("Convertis 100", False, "conversion-incomplete"),
    ("convertis 100 en", False, "conversion-incomplete"),
    ("Genere un mot de passe", False, "mot-de-passe"),
    # ── questions COMPLETES : la porte doit s'effacer ────────────────────
    ("Ecris un poeme sur la mer", False, None),
    ("Fais un plan de dissertation sur le doute", False, None),
    ("raconte moi une blague", False, None),     # sujet sans enjeu
    ("raconte moi une blague sur les poissons", False, None),
    ("Resume la Revolution francaise", False, None),
    ("explique moi la relativite", False, None),
    ("m explique la photosynthese", False, None),
    ("Explique comment ca marche", False, None),
    ("redige un poeme sur la mer", False, None),
    ("Traduis ce texte en anglais", False, None),
    ("Convertis 100 km en miles", False, None),
    ("genere un mot de passe de 16 caracteres", False, None),
    ("Donne trois couleurs primaires separees par des virgules.", False, None),
    ("Calcule le carre de 9", False, None),
    ("Quelle est la capitale de la France ?", False, None),
    ("quel est le prix du petrole", False, None),
    ("Combien font 17 x 23 ?", False, None),
    ("pourquoi le soleil brille t il si fort", False, None),
    ("resous cette integrale", False, None),
    ("Qui es-tu ?", False, None),
    ("Tu es plus fort que ChatGPT ?", False, None),
    ("bonjour, comment vas-tu", False, None),
    # ── une conversation en cours change la lecture ─────────────────────
    ("Ecris un poeme", True, None),      # le sujet est ce dont on parlait
    ("Explique ca", True, None),
    ("Et ensuite ?", True, None),
    ("Continue", True, None),            # verbes sequentiels : objet = tour precedent
    ("developpe", True, None),
    ("traduis ca", True, None),
    ("Explique", True, "verbe-nu"),      # mais « Explique » SEUL reste ambigu
]


@pytest.mark.parametrize("question,contexte,famille", _CAS)
def test_detection(question, contexte, famille):
    besoin = C.detecter(question, a_contexte=contexte)
    if famille is None:
        assert besoin is None, (
            f"« {question} » (contexte={contexte}) est une question COMPLETE "
            f"et a ete declaree trop ouverte : {besoin.famille}")
    else:
        assert besoin is not None, (
            f"« {question} » (contexte={contexte}) est trop ouverte et n'a "
            f"pas ete signalee")
        assert besoin.famille == famille, (besoin.famille, famille)
        assert 1 <= len(besoin.questions) <= 3


def test_le_texte_livre_contient_les_questions():
    besoin = C.detecter("Ecris un poeme", a_contexte=False)
    texte = besoin.texte()
    assert "1." in texte and "2." in texte
    for question in besoin.questions:
        assert question in texte
    # deux questions -> numerotees ; une seule -> pas de « 2. »
    un = C.detecter("Explique").texte()
    assert "1. " in un and "2. " not in un


def test_le_texte_est_en_francais_accentue():
    """Le message part a l'utilisateur : il est redige, pas code."""
    texte = C.detecter("Explique").texte()
    assert "précision" in texte and "réponse" in texte


# ── 2. non-regression sur le banc ------------------------------------------

def test_aucune_question_du_banc_ne_declenche():
    """Le banc 100 est fait de questions COMPLETES.

    Si la porte en interceptait une seule, le score du banc deviendrait
    incomparable (une relance n'est pas une reponse) — et un utilisateur
    qui pose une vraie question recevrait « precise ta demande ».
    """
    import banc_100 as B            # noqa: PLC0415 (import paresseux : chemin)

    ratees = []
    for qid, cat, question, _controle in B.QUESTIONS:
        for contexte in (False, True):
            besoin = C.detecter(question, a_contexte=contexte)
            if besoin is not None:
                ratees.append((qid, cat, besoin.famille))
    assert not ratees, f"le banc est intercepte par la porte : {ratees}"
    assert len(B.QUESTIONS) >= 100


# ── 3. le recollage ---------------------------------------------------------

def test_fusion_par_concatenation():
    """La concatenation EST le recollage : les trois familles en dependent.

    Sur « Convertis 100 » + « en km », une fusion « enrichie » (« Precision :
    en km ») rendrait la phrase incalculable par pal.py — le calcul
    redeviendrait une generation.
    """
    assert C.fusionner("Explique", "la photosynthese") == \
        "Explique la photosynthese"
    assert C.fusionner("Ecris un poeme", "sur la mer") == \
        "Ecris un poeme sur la mer"
    assert C.fusionner("Convertis 100", "en km") == "Convertis 100 en km"
    assert C.fusionner("Explique", "") == "Explique"
    assert C.fusionner("", "la mer") == "la mer"


def test_a_un_sujet_ignore_la_politesse():
    """Le pre-chauffage `executer_detaille("Bonjour")` n'est pas un sujet."""
    tours = [{"role": "user", "content": "Bonjour"},
             {"role": "assistant", "content": "Bonjour ! Comment puis-je..."},
             {"role": "user", "content": "merci beaucoup"}]
    assert C.a_un_sujet(tours) is False
    tours.append({"role": "user", "content": "parle moi des volcans"})
    assert C.a_un_sujet(tours) is True
    assert C.a_un_sujet([]) is False


def test_le_suivi_ne_se_pose_qu_une_fois():
    suivi = C.Suivi()
    assert suivi.en_attente() is False
    suivi.poser("Explique", C.detecter("Explique"))
    assert suivi.en_attente() is True
    fusion = suivi.consommer("la photosynthese")
    assert fusion == "Explique la photosynthese"
    assert suivi.en_attente() is False      # un seul tour, jamais deux
    assert suivi.consommer("autre chose") is None


# ── 4. branchement dans la cascade -----------------------------------------

@pytest.fixture
def ia(monkeypatch, tmp_path):
    """RontoGPT sans GGUF ni reseau, avec une conversation ISOLEE.

    La conversation est isolee parce que la porte la LIT (elle ne demande
    pas de precision quand un sujet vient d'etre donne) : sans ca, le test
    dependrait du `.conversation.jsonl` reel du projet.
    """
    from aura import (llama_cerveau, memoire_web, filtre_instantane,
                      micro_faits, rag_binaire, serveur_lora)
    from aura.memoire_conversation import MemoireConversation
    from aura.orchestrateur import RontoGPT

    appels: list[int] = []

    def _generer(*a, **k):
        appels.append(1)
        return "REPONSE DU CERVEAU"

    monkeypatch.setattr(llama_cerveau, "generer", _generer)
    monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
    monkeypatch.setattr(filtre_instantane, "repondre_cache", lambda q: None)
    monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
    # Les micro-faits repondraient « la photosynthese » : ici on ne mesure que
    # la boucle de clarification et la generation, donc on les met a l'ecart
    # comme les autres couches non generatives (rag, filtre, memoire web).
    monkeypatch.setattr(micro_faits, "resoudre", lambda q: None)
    monkeypatch.setattr(memoire_web, "chercher", lambda *a, **k: "")
    monkeypatch.setattr(serveur_lora, "_actifs", lambda: False)
    instance = RontoGPT()
    instance._historique = MemoireConversation(chemin=tmp_path / "conv.jsonl")
    instance.appels = appels          # type: ignore[attr-defined]
    return instance


class TestBranchement:
    def test_question_ouverte_ne_genere_rien(self, ia):
        """« Explique » : on demande une precision, on ne redige pas."""
        r = ia.executer_detaille("Explique")
        assert r["cerveau_choisi"] == "clarification"
        assert r["en_attente"] is True
        assert r["experts"] == {"clarification"}
        assert not ia.appels, "la porte ne doit PAS appeler le cerveau"
        assert "?" in r["reponse"]
        assert r["clarification"]["famille"] == "verbe-nu"

    def test_le_tour_suivant_est_recolle_puis_repondu(self, ia):
        ia.executer_detaille("Explique")
        r = ia.executer_detaille("la photosynthese")
        assert r["cerveau_choisi"] == f"{llama_cerveau.famille()}-1b"
        assert r["question"] == "Explique la photosynthese"
        assert len(ia.appels) == 1

    def test_un_seul_tour_jamais_de_ping_pong(self, ia):
        """Si la « precision » reste vague, on repond quand meme."""
        ia.executer_detaille("Explique")
        r = ia.executer_detaille("je sais pas trop")
        assert r["cerveau_choisi"] != "clarification"
        assert ia._clarif.en_attente() is False

    def test_les_experts_passent_avant_la_porte(self, ia, monkeypatch):
        """ORDRE : une couche qui repond gagne, meme sur « Explique ».

        C'est l'invariant declare dans `registre_experts.PORTES`
        (« apres-experts »). Une porte posee trop tot refuserait de traiter
        une question que le systeme sait deja trancher — le bug du cache
        semantique, a l'envers.
        """
        from aura import fast_lang
        monkeypatch.setattr(fast_lang.moteur, "repondre",
                            lambda q: "resolu par fast-lang")
        r = ia.executer_detaille("Explique")
        assert r["cerveau_choisi"] == "fast-lang"

    def test_le_niveau0_exact_gagne_aussi(self, ia, monkeypatch):
        from aura import filtre_instantane
        monkeypatch.setattr(filtre_instantane, "repondre",
                            lambda q: "144" if q == "Calcule" else None)
        r = ia.executer_detaille("Calcule")
        assert r["cerveau_choisi"] == "niveau0-instantane"
        assert r["reponse"] == "144"

    def test_la_relance_n_est_pas_une_reponse_en_cache(self, ia, monkeypatch):
        """Une relance ne doit JAMAIS etre memorisee comme une reponse."""
        from aura import filtre_instantane
        enregistres: list[tuple[str, str]] = []
        monkeypatch.setattr(filtre_instantane, "enregistrer",
                            lambda q, r: enregistres.append((q, r)))
        ia.executer_detaille("Explique")
        assert enregistres == []
        ia.executer_detaille("la photosynthese")
        assert any("photosynthese" in q for q, _ in enregistres)

    def test_acquittement_abandonne_la_relance(self, ia):
        """« ok » n'apporte aucune precision : on ne recolle pas « Explique ok »."""
        ia.executer_detaille("Explique")
        r = ia.executer_detaille("ok")
        assert r["cerveau_choisi"] == "constante"
        assert ia._clarif.en_attente() is False

    def test_nouvelle_conversation_oublie_la_relance(self, ia):
        ia.executer_detaille("Explique")
        ia.reinitialiser_conversation()
        assert ia._clarif.en_attente() is False
        r = ia.executer_detaille("la photosynthese")
        assert r["question"] == "la photosynthese"      # aucun recollage

    def test_le_sujet_de_la_conversation_suffit(self, ia):
        """Un tour utilisateur porteur d'un sujet ferme la porte."""
        ia.executer_detaille("parle moi des volcans")
        r = ia.executer_detaille("Ecris un poeme")
        assert r["cerveau_choisi"] != "clarification"

    def test_coupe_circuit_par_variable(self, ia, monkeypatch):
        monkeypatch.setenv("AURA_CLARIFICATION", "0")
        r = ia.executer_detaille("Explique")
        assert r["cerveau_choisi"] == f"{llama_cerveau.famille()}-1b"
        assert len(ia.appels) == 1

    def test_pas_de_porte_quand_un_calcul_est_demande(self, ia):
        """(X, y) fournis : la demande est un calcul, rien a preciser."""
        assert ia._porte_clarification("Explique", [0, 1], [1, 2]) is None
        assert ia._porte_clarification("Explique", None, None) is not None
