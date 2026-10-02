"""Ameliorations v4 : profondeur de raisonnement + bibliotheque etendue.

Leviers valides puis implementes :
- AUTO-THINK : brouillon <thinking> pose par l'orchestrateur sur les
  puzzles et enigmes (llama_cerveau.generer pense aussi sans reflexion
  native, via une instruction dans le TOUR UTILISATEUR — en prefill
  assistant le 1B recite l'instruction au lieu de la suivre) ;
- MAJORITE D'EXPERTS : N essais echantillonnes, CHAQUE valide par un
  solveur deterministe, vote sur la reponse (PoT + mini-SAT) ;
- ARBRE DE REFLEXION (MCTS) : chaque essai devient une branche notee,
  la branche gagnante est reinjectee comme brouillon prouve ;
- concepts.toml : bibliotheque etendue depuis le savoir RAG verifie ;
- REPLI DOMAIN : bande d'indice -> rappel de domaine au lieu du silence.
"""
import time

import numpy as np
import pytest

from aura import concepts, embeddings, orchestrateur as orch
from aura.orchestrateur import Aura1B


# ── 1. Auto-think ───────────────────────────────────────────────────────

class TestAutoThink:
    PILE = "Paul a 3 ans de plus que Marie qui a le double de Leo (4 ans). Quel age a Paul ?"

    def test_puzzle_active_think(self):
        assert Aura1B._auto_think(self.PILE) is True

    def test_enigme_logique_active_think(self):
        assert Aura1B._auto_think(
            "Deux chevaliers et un menteur sont devant toi, qui dit la verite ?"
        ) is True

    @pytest.mark.parametrize("q", [
        "bonjour comment vas tu",
        "quelle est la capitale de l australie",
        "raconte moi une blague sur les poissons",
    ])
    def test_question_neutre_sans_think(self, q):
        assert Aura1B._auto_think(q) is False

    @staticmethod
    def _ia_capture(tmp_path):
        from aura.memoire_conversation import MemoireConversation
        ia = Aura1B()
        ia._historique = MemoireConversation(chemin=tmp_path / "conv.jsonl")
        captures = []
        ia._llama = lambda q, *a, **k: captures.append(k) or "ok"
        return ia, captures

    def test_generer_transmet_think(self, tmp_path):
        ia, captures = self._ia_capture(tmp_path)
        ia._generer("q", "", "", think=True)
        assert captures and captures[-1].get("think") is True

    def test_generer_sans_think_ne_pose_pas_le_kwarg(self, tmp_path):
        # compat : les mocks a signature etroite ne voient PAS le kwarg
        # quand think est inactif (comportement d'avant inchange)
        ia, captures = self._ia_capture(tmp_path)
        ia._generer("q", "", "")
        assert captures and "think" not in captures[-1]

    def test_pipeline_pose_think_sur_puzzle(self, monkeypatch, tmp_path):
        """executer_detaille passe think=True sur un puzzle (et pas ailleurs)."""
        from aura import filtre_instantane, llama_cerveau, rag_binaire
        from aura.memoire_conversation import MemoireConversation
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "n importe")
        monkeypatch.setattr(Aura1B, "_executer_experts",
                            lambda *a, **k: ("", ""))
        monkeypatch.setattr(Aura1B, "_a_besoin_verification",
                            staticmethod(lambda *a, **k: False))
        captures = []

        def faux_generer(self, *args, **kwargs):
            captures.append(kwargs)
            return "reponse de test"

        monkeypatch.setattr(Aura1B, "_generer", faux_generer)

        def _ia():
            ia = Aura1B()
            ia._historique = MemoireConversation(chemin=tmp_path / "c.jsonl")
            return ia

        r1 = _ia().executer_detaille(self.PILE)
        assert r1["reponse"] == "reponse de test"
        assert captures[-1].get("think") is True

        r2 = _ia().executer_detaille("pourquoi le soleil brille t il si fort")
        assert r2["reponse"] == "reponse de test"
        assert not captures[-1].get("think")


class TestThinkCerveau:
    """llama_cerveau.generer : think ouvre la reflexion masquee."""

    @staticmethod
    def _serveur(monkeypatch):
        from aura import serveur_lora
        capture = {}

        def faux_chat(messages, max_tokens=256):
            capture["messages"] = messages
            capture["budget"] = max_tokens
            return "La reponse."

        monkeypatch.setattr(serveur_lora, "_actifs", lambda: True)
        monkeypatch.setattr(serveur_lora, "_actif", (("general", 1.0),))
        monkeypatch.setattr(serveur_lora, "chat", faux_chat)
        return capture

    def test_think_pose_l_instruction_meme_avec_systeme(self, monkeypatch):
        from aura import llama_cerveau, serveur_lora
        capture = self._serveur(monkeypatch)
        r = llama_cerveau.generer("quel age a Paul ?", systeme="SYS",
                                  think=True)
        assert r == "La reponse."
        textes = " ".join(m["content"] for m in capture["messages"])
        assert "<thinking>" in textes
        assert capture["budget"] >= 400          # budget double

    def test_sans_think_aucune_instruction_avec_systeme(self, monkeypatch):
        from aura import llama_cerveau
        capture = self._serveur(monkeypatch)
        llama_cerveau.generer("quel age a Paul ?", systeme="SYS")
        textes = " ".join(m["content"] for m in capture["messages"])
        assert "<thinking>" not in textes

    @staticmethod
    def _cerveau(monkeypatch):
        from aura import llama_cerveau, serveur_lora

        class FauxLlm:
            _n_ctx = 4096

            def __init__(self):
                self.appels = []

            def tokenize(self, texte, add_bos=False):
                return list(texte)

            def create_chat_completion(self, messages, **kw):
                self.appels.append(list(messages))
                return {"choices": [{"message": {"content": "Reponse."}}]}

        faux = FauxLlm()
        monkeypatch.setattr(llama_cerveau, "_charger", lambda: faux)
        monkeypatch.setattr(serveur_lora, "_actifs", lambda: False)
        return faux

    def test_think_in_process_pose_instruction_dans_le_tour_user(
            self, monkeypatch):
        # l'instruction vit dans le tour UTILISATEUR : en prefill
        # assistant, le 1B recite l'instruction (parrot mesure en bench)
        from aura import llama_cerveau
        faux = self._cerveau(monkeypatch)
        llama_cerveau.generer("question", systeme="SYS", think=True)
        messages = faux.appels[-1]
        assert any(m["role"] == "user" and "<thinking>" in m["content"]
                   for m in messages)
        assert not any(m["role"] == "assistant" for m in messages)

    def test_sans_think_in_process_inchange(self, monkeypatch):
        from aura import llama_cerveau
        faux = self._cerveau(monkeypatch)
        llama_cerveau.generer("question", systeme="SYS")
        messages = faux.appels[-1]
        assert not any("<thinking>" in m["content"] for m in messages)


# ── 2. Majorite d'experts + arbre (PoT) ─────────────────────────────────

class TestMajoritePot:
    BON = ("ETAPE 1 : Paul a le double de Marie = 4 * 2\n"
           "ETAPE 2 : Paul a 3 ans de plus = 8 + 3\nREPONSE : 11 ans.")
    # ETAPE 1 doit contenir un OPERATEUR (extraire_etapes exige un calcul)
    BON2 = ("ETAPE 1 : Marie a le double de Leo = 2 * 4\n"
            "ETAPE 2 : Paul a 8 ans puis 3 de plus = 8 + 3\nREPONSE : 11 ans.")
    CASSE = "ETAPE 1 : x = 5 / 0\nETAPE 2 : y = 1 + 1\nREPONSE : ?"

    @staticmethod
    def _lancer(monkeypatch, suite):
        """mock a appels ordonnees + capture de la synthese."""
        it = iter(suite)
        captures = []

        def faux(*a, **k):
            captures.append(k)
            return next(it)

        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer", faux)
        ia = Aura1B()
        return ia, captures

    def test_vote_majoritaire_et_arbre(self, monkeypatch):
        # essai 1 casse, essais 2 et 3 convergent vers 11 -> vote 2/2
        ia, captures = self._lancer(
            monkeypatch, [self.CASSE, self.BON, self.BON2,
                          "Paul a 11 ans."])
        r = ia._resoudre_par_pot("puzzle d ages")
        assert r == "Paul a 11 ans."
        # l'arbre conserve les 3 branches : KO + 2 seriees
        arbre = ia._dernier_arbre
        assert arbre is not None and len(arbre.branches) == 3
        assert arbre.branches[0].serie is False
        assert arbre.branches[1].serie is True
        trace = arbre.trace()
        assert "[KO]" in trace and "[OK]" in trace
        # la synthese recoit la verification + le brouillon prouve
        ctx = captures[-1].get("contexte_web") or ""
        assert "= 11" in ctx
        assert "REFLEXION PROUVEE" in ctx

    def test_divergence_premier_essai_valid_gagne(self, monkeypatch):
        autre = ("ETAPE 1 : Marie a le double de Leo = 4 * 2\n"
                 "ETAPE 2 : Marie a 3 ans de plus = 8 + 3\nREPONSE : 12 ans.")
        ia, captures = self._lancer(
            monkeypatch, [self.BON, autre, self.BON2, "Paul a 11 ans."])
        r = ia._resoudre_par_pot("puzzle d ages")
        assert r == "Paul a 11 ans."
        ctx = captures[-1].get("contexte_web") or ""
        assert "= 11" in ctx and "= 12" not in ctx   # ex aequo -> 1er vu

    def test_majorite_coupée_a_un(self, monkeypatch):
        monkeypatch.setenv("AURA_MAJORITE_N", "1")
        ia, captures = self._lancer(monkeypatch, [self.BON, "Paul a 11 ans."])
        r = ia._resoudre_par_pot("puzzle d ages")
        assert r == "Paul a 11 ans."
        assert len(captures) == 2               # 1 essai + 1 synthese
        assert len(ia._dernier_arbre.branches) == 1

    def test_synthese_sans_valeur_relance_une_fois(self, monkeypatch):
        """La synthese ignore les valeurs verifiees -> relance explicite.

        Mesure en live : le 1B repondait « 7 » alors que le bloc verifiait
        « = 9 » — une relance qui nomme la valeur corrige le tir.
        """
        monkeypatch.setenv("AURA_MAJORITE_N", "2")
        mauvaise = "Alice a 7 ans, enfin je crois."
        ia, captures = self._lancer(
            monkeypatch, [self.BON, self.BON2, mauvaise, "Paul a 11 ans."])
        r = ia._resoudre_par_pot("puzzle d ages")
        assert r == "Paul a 11 ans."
        assert len(captures) == 4          # 2 essais + synthese + relance
        assert "La valeur exacte est 11" in captures[-1]["contexte_web"]

    def test_synthese_obstinee_phrase_du_code(self, monkeypatch):
        """Relance ratee aussi : la phrase AST-verifiee complete la sortie."""
        monkeypatch.setenv("AURA_MAJORITE_N", "2")
        ia, captures = self._lancer(
            monkeypatch, [self.BON, self.BON2, "pas vu", "toujours pas"])
        r = ia._resoudre_par_pot("puzzle d ages")
        assert len(captures) == 4               # 2 essais + 2 synthese
        assert ("Verification exacte : Paul a 3 ans de plus = 8 + 3 = 11."
                in r)
        assert "11" in r                       # la valeur prouvee est la

    def test_aucun_essai_valide_replie(self, monkeypatch):
        ia, captures = self._lancer(
            monkeypatch, [self.CASSE, self.CASSE, self.CASSE])
        assert ia._resoudre_par_pot("puzzle d ages") is None
        assert len(ia._dernier_arbre.branches) == 3


class TestReglagesMajorite:
    def test_nb_borne_et_coupe_budget(self, monkeypatch):
        monkeypatch.setenv("AURA_MAJORITE_N", "9")
        assert orch._nb_majorite() == 3          # plafond
        monkeypatch.setenv("AURA_MAJORITE_N", "0")
        assert orch._nb_majorite() == 1          # coupe
        monkeypatch.setenv("AURA_MAJORITE_N", "abc")
        assert orch._nb_majorite() == 3          # illisible -> defaut
        monkeypatch.delenv("AURA_MAJORITE_N", raising=False)
        monkeypatch.setattr(orch, "_BUDGET_S", 45.0)
        assert orch._nb_majorite(time.time() - 100) == 1   # epuise
        assert orch._nb_majorite(None) == 3                # sans horloge

    def test_cle_de_vote_pot(self):
        assert orch._cle_reponse_pot(
            "ETAPE 1 : x = 1 + 1\nREPONSE : 11 ans.", None) == "11"
        assert orch._cle_reponse_pot("pas de ligne reponse", 7.0) == "7"
        # sans chiffre : tout le texte normalise sert de cle (plus
        # discriminant qu un tronage arbitraire)
        assert orch._cle_reponse_pot(
            "REPONSE : onze reponses possibles", None) == "onze reponses possibles"


# ── 3. Majorite sur les enigmes (mini-SAT) ──────────────────────────────

class TestMajoriteLogique:
    F_ROUGE = ("ENTITES : Anna, Bob\nDOMAINE : rouge, vert\n"
               "TOUS DIFFERENTS\nCONDITION : Anna == rouge")
    F_VERT = ("ENTITES : Anna, Bob\nDOMAINE : rouge, vert\n"
              "TOUS DIFFERENTS\nCONDITION : Anna == vert")

    @staticmethod
    def _lancer(monkeypatch, suite):
        it = iter(suite)
        captures = []

        def faux(q, **kw):
            captures.append(kw)
            return next(it)

        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer", faux)
        return Aura1B(), captures

    def test_vote_deux_sur_trois(self, monkeypatch):
        # 2 formalisations vers Anna=rouge, 1 vers Anna=vert -> gagnant ROUGE
        ia, captures = self._lancer(
            monkeypatch, [self.F_ROUGE, self.F_VERT, self.F_ROUGE,
                          "Anna est rouge."])
        r = ia._resoudre_par_logique("enigme de test")
        assert r == "Anna est rouge."
        assert len(captures) == 4                # 3 formalisations + synthese
        ctx = captures[-1].get("contexte_web") or ""
        assert "- Anna : rouge" in ctx
        assert "- Bob : vert" in ctx

    def test_relance_unique_sur_premier_rejet(self, monkeypatch):
        # AURA_MAJORITE_N=1 : ancien comportement (relance self-debug)
        monkeypatch.setenv("AURA_MAJORITE_N", "1")
        ia, captures = self._lancer(
            monkeypatch, ["n importe quoi", self.F_ROUGE, "Anna est rouge."])
        r = ia._resoudre_par_logique("enigme de test")
        assert r == "Anna est rouge."
        assert len(captures) == 3                # rejet + relance + synthese


# ── 4. concepts.toml (bibliotheque extensible) ──────────────────────────

class TestConceptsToml:
    @staticmethod
    def _biblio(tmp_path, monkeypatch, contenu):
        f = tmp_path / "concepts.toml"
        f.write_text(contenu, encoding="utf-8")
        avant = dict(concepts._CONCEPTS)
        avant_dom = dict(concepts._DOMAINES)
        monkeypatch.setenv("AURA_CONCEPTS_TOML", str(f))
        monkeypatch.setattr(concepts, "_charge_fait", False)
        try:
            yield concepts
        finally:
            concepts._CONCEPTS.clear()
            concepts._CONCEPTS.update(avant)
            concepts._DOMAINES.clear()
            concepts._DOMAINES.update(avant_dom)

    @pytest.fixture
    def biblio(self, tmp_path, monkeypatch):
        yield from self._biblio(
            tmp_path, monkeypatch,
            '[concepts]\n'
            'chimie-bases = "chimie : le symbole de l hydrogene est H, '
            'celui de l oxygene est O."\n'
            'syllogisme = "syllogisme : version surchargee par l utilisateur."\n'
            'test-domaine = { description = "entite de test pour le repli.", '
            'domaine = "psychologie" }\n')

    def test_toml_etend_la_bibliotheque(self, biblio):
        biblio._charger_toml()
        assert "chimie-bases" in concepts._CONCEPTS
        assert concepts._CONCEPTS["chimie-bases"].startswith("chimie :")
        # forme table : description + domaine
        assert concepts._DOMAINES.get("test-domaine") == "psychologie"

    def test_toml_surcharge_un_concept_integre(self, biblio):
        biblio._charger_toml()
        # une cle existante est bien remplacee par celle du fichier
        assert "surchargee par l utilisateur" in concepts._CONCEPTS[
            "syllogisme"]

    def test_idempotent(self, biblio):
        biblio._charger_toml()
        etat = dict(concepts._CONCEPTS)
        biblio._charger_toml()            # deja charge -> rien ne bouge
        assert concepts._CONCEPTS == etat

    def test_fichier_illisible_ignore(self, tmp_path, monkeypatch):
        f = tmp_path / "casse.toml"
        f.write_text("cec n est pas du toml [[", encoding="utf-8")
        avant = dict(concepts._CONCEPTS)
        monkeypatch.setenv("AURA_CONCEPTS_TOML", str(f))
        monkeypatch.setattr(concepts, "_charge_fait", False)
        concepts._charger_toml()          # ne leve pas
        assert concepts._CONCEPTS == avant
        # le flag revient a l'etat initial au demontage (root relira)

    def test_fichier_absent_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AURA_CONCEPTS_TOML",
                           str(tmp_path / "absent.toml"))
        monkeypatch.setattr(concepts, "_charge_fait", False)
        avant = dict(concepts._CONCEPTS)
        concepts._charger_toml()
        assert concepts._CONCEPTS == avant

    @pytest.mark.skipif(not embeddings.modele_present(),
                        reason="modeles/minilm absent")
    def test_cache_recalcule_si_biblio_changee(self, monkeypatch):
        # le cle du cache matrice suit _CONCEPTS (contrat conserve)
        monkeypatch.setattr(concepts, "_cache_matrice", None)
        monkeypatch.setattr(concepts, "_cache_cle", None)
        monkeypatch.setattr(concepts, "_cache_noms", None)
        concepts._vecteurs_concepts()
        assert concepts._cache_cle == tuple(concepts._CONCEPTS.items())


# ── 5. Repli domaine (zéro muet) ────────────────────────────────────────

class TestRepliDomaine:
    """Scores forces (sans modele) : bande d'indice + mot-cle du domaine."""

    @staticmethod
    def _forcer(monkeypatch, score):
        matrice = np.zeros((1, 384), dtype=np.float32)
        matrice[0, 0] = 1.0
        monkeypatch.setattr(concepts, "_vecteurs_concepts",
                            lambda: (["charge-cognitive"], matrice))
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = float(score)
        monkeypatch.setattr(embeddings, "encoder1", lambda texte: vec)

    def test_bande_indice_avec_mot_cle_repli(self, monkeypatch):
        self._forcer(monkeypatch, 0.35)
        hit = concepts.detecter(
            "je vis beaucoup de stress au travail en ce moment et ca me pese")
        assert hit is not None and hit.get("repli") is True
        assert hit["domaine"] == "psychologie"
        bloc = concepts.ancrage(
            "je vis beaucoup de stress au travail en ce moment et ca me pese")
        assert bloc is not None and bloc.startswith("REPLI DOMAIN")

    def test_bande_indice_sans_mot_cle_silence(self, monkeypatch):
        self._forcer(monkeypatch, 0.35)
        assert concepts.detecter(
            "quel est le prix du petrole en ce moment sur les marches"
        ) is None

    def test_au_dessus_du_seuil_hit_normal(self, monkeypatch):
        self._forcer(monkeypatch, 0.45)
        hit = concepts.detecter(
            "je vis beaucoup de stress au travail en ce moment et ca me pese")
        assert hit is not None and not hit.get("repli")
        assert hit["concept"] == "charge-cognitive"

    def test_sous_la_bande_silence(self, monkeypatch):
        self._forcer(monkeypatch, 0.25)
        assert concepts.detecter(
            "je vis beaucoup de stress au travail en ce moment et ca me pese"
        ) is None

    def test_repli_coupe_par_environnement(self, monkeypatch):
        monkeypatch.setenv("AURA_CONCEPTS_REPLI", "0")
        self._forcer(monkeypatch, 0.35)
        assert concepts.detecter(
            "je vis beaucoup de stress au travail en ce moment et ca me pese"
        ) is None

    @pytest.mark.skipif(not embeddings.modele_present(),
                        reason="modeles/minilm absent")
    def test_hors_sujet_reels_sans_repli(self):
        # garde-fou du repli : un hors-sujet mesure reste muet (bande sans
        # mot-cle de domaine) ; l autre est un faux positif AU-DESSUS du
        # seuil (bruit de bibliotheque connu au calibrage -> "seuil
        # conserve") — le REPLI ne doit dans tous les cas JAMAIS s activer
        # sur du hors-sujet.
        assert concepts.detecter("calcule 2 puissance 10 s il te plait") is None
        hit = concepts.detecter(
            "quel temps fait il a paris demain sans pluie")
        assert hit is None or not hit.get("repli")
