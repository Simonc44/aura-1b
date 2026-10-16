"""Tests du RAG binaire (cascade) et de l'ancrage extractif anti-hallucination.

Sources des mecanismes :
- cascade : RAG local (~5 ms) -> web verifie (Knuckles Go) -> graphe -> abstention
- ancrage extractif : une definition CITE les extraits verifies au lieu de
  les faire reformuler par le 1B (zero generation = zero hallucination).
"""
from aura import orchestrateur as orch
from aura import rag_binaire


def _ia_neutre(monkeypatch):
    """RontoGPT sans GGUF ni reseau : la plomberie, pas le contenu.

    Neutralise TOUT ce qui peut toucher disque/reseau : filtre instantane,
    web (les 2 variantes), cerveau, agents couteux (compression, plan,
    double passe). Les tests activent ce qu'ils veulent observer.
    """
    from aura import agents, filtre_instantane, llama_cerveau, memoire_web
    monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
    monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
    monkeypatch.setattr(memoire_web, "chercher", lambda q, **k: "")
    monkeypatch.setattr(llama_cerveau, "generer",
                        lambda *a, **k: "REponse du cerveau")
    # agents : pass-through sans LLM
    monkeypatch.setattr(agents, "compresser", lambda texte, q, **k: texte)
    monkeypatch.setattr(agents, "en_triplets", lambda texte, q: texte)
    monkeypatch.setattr(agents, "planifier", lambda q: "")
    monkeypatch.setattr(agents, "exemple_style", lambda categorie: "")
    return orch.RontoGPT()


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
    def test_definition_source_pertinente_synthetise(self, monkeypatch):
        """Source FORTE (le sujet y figure) -> synthese ancree par le 1B.

        Routage hybride (8 oct. 2026) : quand la source parle vraiment du
        sujet, on redige a partir du VERIFIE au lieu de citer brut.
        """
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.memoire_web, "chercher",
                            lambda q, **k: "Le theoreme de Pythagore relie les cotes d'un triangle rectangle.")
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: "Le theoreme de Pythagore relie les cotes d'un triangle rectangle.")
        r = ia.executer_detaille("qu'est-ce que le theoreme de pythagore ?")
        assert r["cerveau_choisi"] == "def-web+1b"
        assert "Pythagore" in r["reponse"]
        assert "sources verifiees" not in r["reponse"]

    def test_definition_source_hors_sujet_cite(self, monkeypatch):
        """Source FAIBLE (le sujet n'y figure pas) -> citation, PAS de 1B.

        Cas mesure : « fonction canonique » face a des extraits parlant de
        « forme canonique » — le 1B inventait une fausse definition s'il
        redigeait, donc on cite.
        """
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.memoire_web, "chercher",
                            lambda q, **k: "Toute fonction du second degre possede une forme canonique a(x-alpha)^2+beta.")
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("source hors-sujet : pas de generation")))
        r = ia.executer_detaille("c'est quoi une fonction canonique")
        assert r["cerveau_choisi"] == "extraction-web"
        assert "sources verifiees" in r["reponse"]

    def test_citation_ne_finit_pas_en_pleine_phrase(self, monkeypatch):
        """Snippet tronque -> on recule a la derniere phrase complete.

        Un moteur renvoie des extraits coupes net (« ... par a : »), ce qui
        donne l'impression que la reponse s'arrete au milieu.
        """
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(
            orch.memoire_web, "chercher",
            lambda q, **k: ("Titre : X | Extrait : Toute fonction est definie "
                            "sur un ensemble. Chacune des expressions ne "
                            "differe que par une factorisation par a :"))
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("pas de generation")))
        r = ia.executer_detaille("c'est quoi une fonction canonique")
        assert "sources verifiees" in r["reponse"]
        assert "factorisation" not in r["reponse"]     # fragment coupe retire
        assert r["reponse"].rstrip().endswith(".")

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

        L'important est qu'elle passe par le CERVEAU, pas par l'ancrage
        extractif : citer des sources n'a aucun sens pour une question
        creative.
        """
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.memoire_web, "chercher", lambda q, **k: "fait utile")
        appel = {}
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: appel.setdefault("simple", "GEN"))
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

    def test_explique_pourquoi_reste_un_developpement(self, monkeypatch):
        """« Explique pourquoi X » = DEVELOPPEMENT, jamais citation-extraction.

        Bug deterministe du banc maison (11 oct. 2026) : le gate de l'ancrage
        extractif testait `est_def`, qui couvre « explique » — la question
        partait donc en `extraction-web` et recevait « Aucune source ne
        definit directement « ... » » la ou l'utilisateur attendait un
        developpement. Regle : seuls les LOOKUPS (cf. _RE_DEF_LOOKUP) sont
        cites/extraits ; « explique » reste un developpement, ancrable par le
        web mais genere par le cerveau.
        """
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(
            orch.memoire_web, "chercher",
            lambda q, **k: "Les oceans absorbent une grande part de la chaleur.")
        appels = []
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: appels.append(1) or "Developpement")
        r = ia.executer_detaille(
            "Explique pourquoi les oceans sont essentiels a l'equilibre climatique")
        assert appels, "un developpement doit passer par le cerveau"
        assert r["cerveau_choisi"] != "extraction-web"
        assert "Aucune source ne definit" not in r["reponse"]


# ── 3. Canalisation : le 1B RECLAME le contexte au lieu de repondre ────


class TestCanalisationParAncrage:
    """Mesure du 12 oct. 2026 — contexte web de 3535 car. POURTANT joint :

        « J'ai besoin des informations pour repondre a ta question. Dis-moi :
          « Qu'est-ce qui est fourni ci-dessous et est pertinent » (WEB / BASE
          DE FAITS) ... dis-moi ce que c'est. »

    Question posee : « Qui a gagne le tournoi de tennis de Roland-Garros en
    2025 ? ». 82 s de calcul pour RENDRE LA QUESTION a l'utilisateur, alors
    que les extraits verifies etaient joints. Ici : le bavardage ne part pas,
    et les extraits prennent sa place — zero generation, donc zero nouvelle
    occasion d'inventer.
    """

    FUITE = ("J'ai besoin des informations pour repondre a ta question. "
             "Dis-moi : « Qu'est-ce qui est fourni ci-dessous et est "
             "pertinent » (WEB / BASE DE FAITS) ... dis-moi ce que c'est.")
    CONTEXTE = ("Titre : Canberra | Extrait : Canberra est la capitale de "
                "l'Australie | Source : https://exemple.fr/australie")

    def test_bavardage_remplace_par_les_extraits(self, monkeypatch):
        ia = _ia_neutre(monkeypatch)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(orch.memoire_web, "chercher",
                            lambda q, **k: self.CONTEXTE)
        monkeypatch.setattr(orch.llama_cerveau, "generer",
                            lambda *a, **k: self.FUITE)
        r = ia.executer_detaille("qu'est-ce que la capitale de l'Australie ?")
        rep = r["reponse"]
        assert "dis-moi" not in rep.lower()
        assert "BASE DE FAITS" not in rep
        assert "Canberra" in rep          # les extraits, eux, partent bien
        assert len(rep) > 20              # jamais une reponse vide en sortie
