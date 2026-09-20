"""Tests des 3 nouveaux mecanismes (inspires de GitHub) :
1. Program-of-Thoughts (aura/raisonneur.py) : le 1B ecrit, l'AST verifie
2. MiniRAG-lite (aura/graphe_faits.py) : triplets de faits verifies
3. StoryWriter-lite : redaction section par section dans generer_riche
"""
import pytest

from aura import raisonneur
from aura.orchestrateur import Aura1B


# ── 1. Program-of-Thoughts ──────────────────────────────────────────────

class TestPotDetection:
    def test_puzzle_ages_detecte(self):
        q = ("Paul a 3 ans de plus que Marie. Marie a le double de l'age "
             "de Leo qui a 4 ans. Quel age a Paul ?")
        assert raisonneur.est_puzzle(q)

    def test_question_normale_pas_un_puzzle(self):
        assert not raisonneur.est_puzzle("capitale de l australie")
        assert not raisonneur.est_puzzle("explique le soleil")
        assert not raisonneur.est_puzzle("calcule le carre de 9")

    def test_indices_sans_question_finale_pas_un_puzzle(self):
        assert not raisonneur.est_puzzle("les ages de la population")


class TestPotExtraction:
    BRUT_VALIDE = (
        "ETAPE 1 : Leo a 4 ans, Marie a le double = 2 * 4\n"
        "ETAPE 2 : Paul a 3 ans de plus que Marie = 8 + 3\n"
        "REPONSE : Paul a 11 ans."
    )

    def test_extraire_etapes(self):
        etapes = raisonneur.extraire_etapes(self.BRUT_VALIDE)
        assert len(etapes) == 2
        assert etapes[0][1] == "2 * 4"
        assert etapes[1][1] == "8 + 3"

    def test_verifier_calculs(self):
        etapes = raisonneur.extraire_etapes(self.BRUT_VALIDE)
        res = raisonneur.verifier_calculs(etapes)
        assert res == [8.0, 11.0]

    def test_verifier_calculs_refuse_n_importe_quoi(self):
        etapes = [("x", "2 ** 10000")]
        with pytest.raises(ValueError):
            raisonneur.verifier_calculs(etapes)

    def test_construire_verification(self):
        etapes = raisonneur.extraire_etapes(self.BRUT_VALIDE)
        res = raisonneur.verifier_calculs(etapes)
        bloc = raisonneur.construire_verification(etapes, res)
        assert "VERIFICATION" in bloc and "= 8" in bloc and "= 11" in bloc


class TestPotPipeline:
    def test_resoudre_par_pot_valide(self, monkeypatch):
        from aura import llama_cerveau
        reponses = iter([
            "ETAPE 1 : Marie a le double = 2 * 4\n"
            "ETAPE 2 : Paul a 3 de plus = 8 + 3\nREPONSE : 11 ans.",
            "Paul a 11 ans.",
        ])
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: next(reponses))
        r = Aura1B()._resoudre_par_pot("puzzle quelconque")
        assert r == "Paul a 11 ans."

    def test_resoudre_par_pot_replie_si_pas_d_etapes(self, monkeypatch):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "je ne sais pas vraiment")
        assert Aura1B()._resoudre_par_pot("puzzle") is None

    def test_resoudre_par_pot_replie_si_calcul_casse(self, monkeypatch):
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k:
                            "ETAPE 1 : x = 5 / 0\nETAPE 2 : y = 1 + 1\n")
        assert Aura1B()._resoudre_par_pot("puzzle") is None

    def test_pipeline_pot_complete(self, monkeypatch):
        """Question puzzle sans web ni donnees -> cerveau pot-1b+ast."""
        from aura import filtre_instantane, llama_cerveau
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda q, contexte_web="", formule="",
                            max_tokens=None, historique=None, systeme=None,
                            contexte_faits="":
                            "ETAPE 1 : Marie = 2 * 4\n"
                            "ETAPE 2 : Paul = 8 + 3\nREPONSE : 11 ans."
                            if systeme else "Paul a 11 ans.")
        ia = Aura1B()
        r = ia.executer_detaille(
            "Paul a 3 ans de plus que Marie qui a le double de Leo (4 ans). "
            "Quel age a Paul ?")
        assert r["cerveau_choisi"] == "pot-1b+ast"
        assert r["reponse"] == "Paul a 11 ans."
        assert "pot" in r["experts"]


# ── 2. MiniRAG-lite (graphe de faits) ───────────────────────────────────

class TestGrapheFaits:
    @pytest.fixture(autouse=True)
    def _graphe_vierge(self, monkeypatch, tmp_path):
        from aura import graphe_faits
        monkeypatch.setattr(graphe_faits, "_FICHIER", str(tmp_path / "g.jsonl"))
        monkeypatch.setattr(graphe_faits, "_chargé", True)
        monkeypatch.setattr(graphe_faits, "_triplets", [])
        monkeypatch.setattr(graphe_faits, "_index", {})
        self.gf = graphe_faits

    def test_ajouter_et_chercher(self):
        self.gf.ajouter("Canberra", "capitale de", "l'Australie")
        ctx = self.gf.chercher("quelle est la capitale de l australie")
        assert "Canberra" in ctx

    def test_deduplication(self):
        assert self.gf.ajouter("X", "est", "Y")
        assert not self.gf.ajouter("X", "est", "Y")

    def test_apprentissage_exige_verification_web(self):
        # reponse NON verifiee : rien n'apprend
        assert self.gf.apprendre_de_reponse(
            "Sydney est la capitale de l'Australie.", verifiee_web=False) == 0
        # reponse verifiee : extraction conservatrice
        n = self.gf.apprendre_de_reponse(
            "Canberra est la capitale de l'Australie.", verifiee_web=True)
        assert n == 1
        ctx = self.gf.chercher("capitale de l australie")
        assert "Canberra" in ctx

    def test_chercher_vide_si_rien_de_pertinent(self):
        self.gf.ajouter("Canberra", "capitale de", "l'Australie")
        assert self.gf.chercher("recette de pain au chocolat") == ""

    def test_extraire_patterns(self):
        s, r, o = self.gf.extraire_de_reponse(
            "Emmanuel Macron est le president de la France.")[0]
        assert "Macron" in s and "France" in o


# ── 3. StoryWriter-lite (redaction section par section) ─────────────────

class TestStoryWriterLite:
    PLAN = ("PARTIE 1 : Etat des lieux du solaire\n"
            "PARTIE 2 : L'essor de l'eolien\n"
            "PARTIE 3 : Comparaison et perspectives\n"
            "CONNECTEURS : Neanmoins, Par consequent")

    def _llm_sections(self, textes):
        class FauxLlm:
            def __init__(self):
                self.appels = []

            def create_chat_completion(self, **kw):
                self.appels.append(kw)
                contenu = textes[len(self.appels) - 1]
                return {"choices": [{"message": {"content": contenu}}]}
        return FauxLlm()

    def test_redaction_par_sections(self, monkeypatch):
        from aura import llama_cerveau as lc
        # appel 1 = le plan, puis les 3 sections
        llm = self._llm_sections([self.PLAN, "SECTION UN.",
                                  "SECTION DEUX.", "TROIS."])
        monkeypatch.setattr(lc, "_charger", lambda: llm)
        monkeypatch.setattr(lc, "grammaire_plan", lambda: None)
        r = lc.generer_riche("Redige une analyse comparee du solaire et "
                             "de l'eolien en France")
        # 3 sections + 1 passe plan = 4 appels
        assert len(llm.appels) == 4
        assert "SECTION UN." in r and "TROIS." in r
        # la 2e section a recu la memoire de la 1re (coherence)
        prompt_sec2 = llm.appels[2]["messages"][0]["content"]
        assert "SECTION UN." in prompt_sec2

    def test_repli_passe_unique_si_une_section_echoue(self, monkeypatch):
        from aura import llama_cerveau as lc

        class FauxLlm:
            def __init__(self):
                self.n = 0

            def create_chat_completion(self, **kw):
                self.n += 1
                if self.n == 1:
                    return {"choices": [{"message": {"content": self.PLAN}}]}
                if self.n in (2, 3):
                    raise RuntimeError("boom")
                return {"choices": [{"message": {"content": "PASSE UNIQUE."}}]}

        llm = FauxLlm()
        monkeypatch.setattr(lc, "_charger", lambda: llm)
        monkeypatch.setattr(lc, "grammaire_plan", lambda: None)
        monkeypatch.setattr(lc, "generer", lambda *a, **k: "REPLI-SIMPLE")
        r = lc.generer_riche("Redige une analyse comparee du solaire "
                             "et de l'eolien en France")
        assert r == "REPLI-SIMPLE"
