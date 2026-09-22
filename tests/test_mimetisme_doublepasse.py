"""Tests des agents 6 (mimétisme few-shot) et 7 (double passe)."""
from aura import agents


class TestMimetisme:
    def test_brique_par_categorie(self):
        for cle in ("general", "web", "math"):
            brique = agents.exemple_style("question", categorie=cle)
            assert brique and "Exemples de style" in brique

    def test_brique_devinee_maths(self):
        assert agents.exemple_style("combien fait 12 fois 12") is not None

    def test_desactive_si_agents_off(self, monkeypatch):
        monkeypatch.setattr(agents, "actifs", lambda: False)
        assert agents.exemple_style("quoi que ce soit") is None

    def test_brique_contient_une_consigne(self):
        brique = agents.exemple_style("question", categorie="math")
        assert "Rédige" in brique


class TestDoublePasse:
    def test_texte_court_inchange(self, monkeypatch):
        monkeypatch.setattr(agents, "actifs", lambda: True)
        assert agents.double_passe("trop court", "q") == "trop court"

    def test_inactif_renvoie_le_texte(self, monkeypatch):
        monkeypatch.setattr(agents, "actifs", lambda: False)
        long = "x" * 200
        assert agents.double_passe(long, "q") == long

    def test_critique_rien_garde_le_texte(self, monkeypatch):
        monkeypatch.setattr(agents, "actifs", lambda: True)
        texte = "x" * 120

        class FauxLlm:
            def create_chat_completion(self, **k):
                return {"choices": [{"message": {"content": "RIEN"}}]}

        import aura.llama_cerveau as lc
        monkeypatch.setattr(lc, "_charger", lambda: FauxLlm())
        assert agents.double_passe(texte, "q") == texte

    def test_reformulation_appliquee_si_credential(self, monkeypatch):
        monkeypatch.setattr(agents, "actifs", lambda: True)
        texte = "Phrase un peu molle. " * 12  # ~264 car
        appels = []

        class FauxLlm:
            def create_chat_completion(self, **k):
                appels.append(k)
                contenu = k["messages"][0]["content"]
                if "correcteur" in contenu:
                    return {"choices": [{"message": {"content":
                        "- repetition\n- formulation molle"}}]}
                return {"choices": [{"message": {"content":
                        "Version corrigee, plus nette et sans repetition. "
                        "Chaque idee avance clairement."}}]}

        import aura.llama_cerveau as lc
        monkeypatch.setattr(lc, "_llm", FauxLlm())  # cache du cerveau
        monkeypatch.setattr(lc, "_charger", lambda: lc._llm)
        resultat = agents.double_passe(texte, "q")
        assert resultat.startswith("Version corrigee")
        assert len(appels) == 2  # critique puis reformulation

    def test_reformulation_trop_courte_rejetee(self, monkeypatch):
        monkeypatch.setattr(agents, "actifs", lambda: True)
        texte = "Phrase correcte. " * 20  # ~280 car

        class FauxLlm:
            def create_chat_completion(self, **k):
                contenu = k["messages"][0]["content"]
                if "correcteur" in contenu:
                    return {"choices": [{"message": {"content":
                            "- un defaut"}}]}
                return {"choices": [{"message": {"content": "Court."}}]}

        import aura.llama_cerveau as lc
        monkeypatch.setattr(lc, "_charger", lambda: FauxLlm())
        assert agents.double_passe(texte, "q") == texte

    def test_cerveau_indisponible_renvoie_le_texte(self, monkeypatch):
        monkeypatch.setattr(agents, "actifs", lambda: True)
        import aura.llama_cerveau as lc
        def boom():
            raise RuntimeError("pas de gguf")
        monkeypatch.setattr(lc, "_charger", boom)
        texte = "y" * 150
        assert agents.double_passe(texte, "q") == texte
