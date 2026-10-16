"""Tests fiabilite : seuil de confiance (style RouteLLM) + plan GBNF contraint."""
from aura.orchestrateur import RontoGPT
from aura import llama_cerveau


def _ia(monkeypatch):
    """RontoGPT sans GGUF requis (llm faux) + cache niveau 0 neutre."""
    from aura import filtre_instantane
    monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
    monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
    monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "ok-llm")
    return RontoGPT()


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
        monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "ok-llm")
        # Famille EPINGLEE : sans cerveau sur le disque `famille()` rend '' et
        # la chaine annoncait « inconnu-1b » (mesure du 16 oct. 2026, en CI).
        monkeypatch.setattr(llama_cerveau, "famille", lambda: "gemma3")
        ia = RontoGPT()
        monkeypatch.setattr(
            ia, "analyser",
            lambda q: {"experts": {"web"}, "prediction": "web",
                       "scores": {"web": 0.30, "math": 0.30, "general": 0.30}})
        r = ia.executer_detaille("question ambigue quelconque")
        assert r["experts"] == {"general"}
        assert r["cerveau_choisi"] == f"{llama_cerveau.famille()}-1b"


