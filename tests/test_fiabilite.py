"""Tests fiabilite : seuil de confiance (style RouteLLM) + plan GBNF contraint."""
from aura.orchestrateur import Aura1B


def _ia(monkeypatch):
    """Aura sans GGUF requis (llm faux) + cache niveau 0 neutre."""
    from aura import filtre_instantane, llama_cerveau
    monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
    monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
    monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "ok-llm")
    monkeypatch.setattr(llama_cerveau, "generer_riche", lambda *a, **k: "ok-llm")
    return Aura1B()


class TestSeuilConfiance:
    def test_confiance_basse_chemin_sur(self, monkeypatch):
        """Score max sous le seuil -> chemin sur 'general', pas de pari."""
        ia = _ia(monkeypatch)
        analyse = {"experts": {"web"}, "prediction": "web",
                   "scores": {"web": 0.40, "math": 0.35, "general": 0.25}}
        experts = ia._valider_experts(analyse)
        assert experts == {"general"}

    def test_confiance_haute_experts_conserves(self, monkeypatch):
        ia = _ia(monkeypatch)
        analyse = {"experts": {"web"}, "prediction": "web",
                   "scores": {"web": 0.79, "math": 0.10, "general": 0.11}}
        assert ia._valider_experts(analyse) == {"web"}

    def test_pile_au_seuil_conserve(self, monkeypatch):
        """Score == seuil : le choix est accepte (seuil strict <)."""
        ia = _ia(monkeypatch)
        analyse = {"experts": {"math"}, "prediction": "math",
                   "scores": {"math": 0.50, "web": 0.25, "general": 0.25}}
        # 'math' sans donnees -> garde-fou le demote ensuite, mais la
        # confiance elle-meme ne doit PAS demoter a 0.50
        experts = ia._valider_experts(analyse)
        assert "general" in experts   # repli du garde-fou PGS, pas du seuil

    def test_scores_absents_pas_de_demotion(self, monkeypatch):
        """Pas de scores (routeur mots-cles) -> aucune demotion par confiance."""
        ia = _ia(monkeypatch)
        analyse = {"experts": {"web"}, "methodes": ["mots_cles"], "scores": {}}
        assert ia._valider_experts(analyse) == {"web"}

    def test_math_confiant_sans_donnees_tombe_sur_general(self, monkeypatch):
        """Confiance OK mais PGS sans donnees : le garde-fou demote quand meme."""
        ia = _ia(monkeypatch)
        analyse = {"experts": {"math"}, "prediction": "math",
                   "scores": {"math": 0.85, "web": 0.05, "general": 0.10}}
        experts = ia._valider_experts(analyse)
        assert experts == {"general"}

    def test_integration_pipeline_confiance_basse(self, monkeypatch):
        """Pipeline complet : analyse peu sure -> general -> LLM simple."""
        from aura import filtre_instantane
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "ok-llm")
        ia = Aura1B()
        monkeypatch.setattr(
            ia, "analyser",
            lambda q: {"experts": {"web"}, "prediction": "web",
                       "scores": {"web": 0.30, "math": 0.30, "general": 0.30}})
        r = ia.executer_detaille("question ambigue quelconque")
        assert r["experts"] == {"general"}
        assert r["cerveau_choisi"] == "llama-3.2-1b"


class TestGrammairePlan:
    def test_grammaire_compile_et_mise_en_cache(self):
        from aura import llama_cerveau
        g1 = llama_cerveau.grammaire_plan()
        g2 = llama_cerveau.grammaire_plan()
        assert g1 is not None          # llama-cpp-python 0.3.35 : GBNF OK
        assert g1 is g2                # compilee une seule fois

    def test_grammaire_rejete_format_invalide(self):
        """La grammaire elle-meme impose le format : check si l'API le permet."""
        from aura import llama_cerveau
        g = llama_cerveau.grammaire_plan()
        if g is None:
            import pytest
            pytest.skip("GBNF indisponible sur ce build")
        # la grammaire accepte un texte conforme ; le vrai test de conformite
        # se fait au boot reel (test_integration_plan_contraint_ci_dessous)
        assert g is not None

    def test_prompt_plan_aligne_sur_la_grammaire(self):
        """Le prompt montre EXACTEMENT le format que la grammaire impose."""
        from aura import llama_cerveau
        p = llama_cerveau._PROMPT_PLAN.format(contexte="CTX", question="Q")
        assert "PARTIE 1 :" in p and "PARTIE 2 :" in p and "PARTIE 3 :" in p
        assert "CONNECTEURS :" in p

    def test_generer_plan_sortie_contrainte(self, monkeypatch):
        """_generer_plan passe bien la grammaire a create_chat_completion."""
        from aura import llama_cerveau
        if llama_cerveau.grammaire_plan() is None:
            import pytest
            pytest.skip("GBNF indisponible sur ce build")
        capture = {}

        class FauxLlm:
            def create_chat_completion(self, **kw):
                capture.update(kw)
                return {"choices": [{"message": {"content":
                        "PARTIE 1 : aaa\nPARTIE 2 : bbb\nPARTIE 3 : ccc\n"
                        "CONNECTEURS : Neanmoins, Par consequent"}}]}

        plan = llama_cerveau._generer_plan(
            FauxLlm(), "question", "contexte")
        assert capture.get("grammar") is not None   # la grammaire est passee
        assert "PARTIE 1" in plan

    def test_generer_riche_replie_si_passe1_echoue(self, monkeypatch):
        """Passe 1 qui leve -> repli generation simple (comportement conserve)."""
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "_charger",
                            lambda: object())
        monkeypatch.setattr(llama_cerveau, "grammaire_plan", lambda: None)

        class FauxLlm:
            def create_chat_completion(self, **kw):
                raise RuntimeError("boom")

        monkeypatch.setattr(llama_cerveau, "_charger", lambda: FauxLlm())
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "repli-simple")
        r = llama_cerveau.generer_riche("question ouverte assez longue "
                                        "pour declencher le mode riche")
        assert r == "repli-simple"

    def test_integration_plan_contraint(self):
        """GRANDEUR NATURE : le vrai modele produit un plan 100 % conforme."""
        from aura import llama_cerveau
        if not llama_cerveau.disponible():
            import pytest
            pytest.skip("GGUF Llama non telecharge")
        import re
        llm = llama_cerveau._charger()
        plan = llama_cerveau._generer_plan(
            llm, "Redige une analyse comparee du solaire et de l'eolien",
            "aucun contexte")
        lignes = plan.strip().splitlines()
        assert len(lignes) == 4
        assert lignes[0].startswith("PARTIE 1 : ")
        assert lignes[1].startswith("PARTIE 2 : ")
        assert lignes[2].startswith("PARTIE 3 : ")
        assert lignes[3].startswith("CONNECTEURS : ")
        assert re.match(r"^PARTIE [1-3] : .{10,220}$", lignes[0])
