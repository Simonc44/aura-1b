"""Tests du RAG binaire (cascade) et de l'ancrage extractif anti-hallucination.

Sources des mecanismes :
- cascade : RAG local (~5 ms) -> web verifie (Knuckles Go) -> graphe -> abstention
- ancrage extractif : une definition CITE les extraits verifies au lieu de
  les faire reformuler par le 1B (zero generation = zero hallucination).
"""
from aura import orchestrateur as orch
from aura import rag_binaire


def _ia_neutre(monkeypatch):
    """Aura sans GGUF ni reseau : la plomberie, pas le contenu.

    Neutralise TOUT ce qui peut toucher disque/reseau : filtre instantane,
    web (les 2 variantes), cerveau, agents couteux (compression, plan,
    double passe). Les tests activent ce qu'ils veulent observer.
    """
    from aura import agents, filtre_instantane, llama_cerveau, memoire_web
    monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
    monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
    monkeypatch.setattr(memoire_web, "chercher", lambda q, **k: "")
    monkeypatch.setattr(memoire_web, "chercher_enrichi", lambda q, **k: "")
    monkeypatch.setattr(llama_cerveau, "generer",
                        lambda *a, **k: "REponse du cerveau")
    monkeypatch.setattr(llama_cerveau, "generer_riche",
                        lambda *a, **k: "REponse du cerveau (riche)")
    # agents : pass-through sans LLM
    monkeypatch.setattr(agents, "compresser", lambda texte, q, **k: texte)
    monkeypatch.setattr(agents, "en_triplets", lambda texte, q: texte)
    monkeypatch.setattr(agents, "planifier", lambda q: "")
    monkeypatch.setattr(agents, "double_passe", lambda texte, q, **k: texte)
    monkeypatch.setattr(agents, "exemple_style", lambda categorie: "")
    return orch.Aura1B()


# ── 1. Cascade RAG binaire ─────────────────────────────────────────────

class TestCascadeRagBinaire:
    def test_hit_rag_court_circuite_web(self, monkeypatch):
        """RAG hit -> reponse extraite, PAS de requete web, PAS de LLM."""
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: "Paris | capitale | France")
        monkeypatch.setattr(orch.memoire_web, "chercher",
                            lambda q, **k: (_ for _ in ()).throw(
                                AssertionError("web ne doit pas etre consulte")))
        r = ia.executer_detaille("qu'est-ce que la capitale de la France ?")
        assert r["cerveau_choisi"] == "rag-binaire"
        assert "Paris" in r["reponse"]

    def test_miss_rag_continue_cascade(self, monkeypatch):
        """RAG miss -> la cascade continue (web consulte)."""
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.memoire_web, "chercher",
                            lambda q, **k: "Paris est la capitale de la France.")
        r = ia.executer_detaille("qu'est-ce que la capitale de la France ?")
        assert r["cerveau_choisi"] != "rag-binaire"

    def test_erreur_rag_ne_bloque_pas(self, monkeypatch):
        """RAG leve -> degrade en miss, cascade continue."""
        ia = _ia_neutre(monkeypatch)
        def boom(q):
            raise RuntimeError("index absent")
        monkeypatch.setattr(rag_binaire, "chercher", boom)
        monkeypatch.setattr(orch.memoire_web, "chercher", lambda q, **k: "fait")
        r = ia.executer_detaille("qu'est-ce que X ?")
        assert r["reponse"]  # une reponse existe quand meme

    def test_rag_inactif_en_mode_riche(self, monkeypatch):
        """Mode riche (analyse) : le RAG ne court-circuite pas."""
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher",
                            lambda q: (_ for _ in ()).throw(
                                AssertionError("riche : pas de RAG")))
        # >= 9 mots sans definition -> mode riche
        ia.executer_detaille(
            "analyse en profondeur les causes economiques sociales et "
            "culturelles de la revolution industrielle du dix-neuvieme siecle")
        # pas d'exception = RAG bien evite


# ── 2. Ancrage extractif des definitions ──────────────────────────────

class TestAncrageExtractif:
    def test_definition_cite_le_web(self, monkeypatch):
        """Definition + web -> le contexte VERIFIE est cite tel quel."""
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.memoire_web, "chercher",
                            lambda q, **k: "Le theoreme de Pythagore relie les cotes d'un triangle rectangle.")
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("le 1B ne doit PAS generer")))
        r = ia.executer_detaille("qu'est-ce que le theoreme de pythagore ?")
        assert "Pythagore" in r["reponse"]
        assert "sources verifiees" in r["reponse"]

    def test_phrase_min_ancree_aussi(self, monkeypatch):
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.memoire_web, "chercher",
                            lambda q, **k: "La theorie de la relativite lie espace et temps.")
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("pas de generation")))
        r = ia.executer_detaille("qu'est-ce que e=mc2")
        assert "relativite" in r["reponse"]

    def test_pas_de_definition_pas_dancrage(self, monkeypatch):
        """Question non-definition : chemin normal (generation).

        Une question creative peut etre routee riche (generer_riche) ou
        simple (generer) : l'important est qu'elle passe par le cerveau,
        PAS par l'ancrage extractif.
        """
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.memoire_web, "chercher", lambda q, **k: "fait utile")
        appel = {}
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: appel.setdefault("simple", "GEN"))
        monkeypatch.setattr(orch.llama_cerveau, "generer_riche",
                            lambda *a, **k: appel.setdefault("riche", "GEN"))
        r = ia.executer_detaille("redige un poeme sur la mer")
        assert appel, "une question non-definition doit passer par le cerveau"
        assert r["reponse"] == "GEN" or r["reponse"].startswith("GEN")

    def test_def_sans_web_n_invente_pas(self, monkeypatch):
        """Definition sans reseau : abstention honnete, pas d'invention."""
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("rien a generer sans contexte")))
        r = ia.executer_detaille("qu'est-ce que la quantification geometrique ?")
        rep = r["reponse"].lower()
        assert ("pas d'acces" in rep or "ne sait" in rep
                or "verifie" in rep or len(r["reponse"]) < 400)
