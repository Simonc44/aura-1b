"""Tests des 3 etapes d'amelioration : lexique, CoT masque, multi-pass."""
from aura import llama_cerveau, memoire_web
from aura.orchestrateur import Aura1B


class TestEtape1Lexique:
    def test_extraire_lexique(self):
        textes = ("Le photovoltaique connait une croissance rapide. "
                  "Les turbines eoliennes offshores progressent. "
                  "Le photovoltaique reste competitif face aux turbines.")
        lexique = memoire_web._extraire_lexique(textes, max_mots=5)
        assert "photovoltaique" in lexique
        assert "turbines" in lexique
        assert len(lexique) <= 5

    def test_chercher_enrichi_sections(self, monkeypatch):
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda q, max_resultats=3, timeout=10:
                            "Titre : t | Extrait : le photovoltaique progresse")
        ctx = memoire_web.chercher_enrichi("le photovoltaique en 2026")
        assert "FAITS WEB" in ctx
        assert "ANALYSES" in ctx
        assert "Mots-cles pertinents" in ctx

    def test_chercher_enrichi_vide_offline(self, monkeypatch):
        monkeypatch.setattr(memoire_web, "chercher", lambda *a, **k: "")
        assert memoire_web.chercher_enrichi("question") == ""


class TestEtape2CoT:
    def test_separer_reflexion_complete(self):
        brut = "<thinking>1. Analyse : ok. 2. Plan : A, B.</thinking>Reponse finale claire."
        propre, reflexion = llama_cerveau.separer_reflexion(brut)
        assert propre == "Reponse finale claire."
        assert "Analyse" in reflexion

    def test_balise_orpheline(self):
        brut = "<thinking>reflexion coupee... Reponse partielle."
        propre, reflexion = llama_cerveau.separer_reflexion(brut)
        assert "<thinking>" not in propre
        assert reflexion == ""  # pas de balise fermante -> reflexion jetee

    def test_sans_thinking(self):
        propre, reflexion = llama_cerveau.separer_reflexion("Reponse simple.")
        assert propre == "Reponse simple."
        assert reflexion == ""


class TestEtape3Multipass:
    def test_prompts_contiennent_les_rails(self):
        p = llama_cerveau._PROMPT_PLAN.format(contexte="CTX", question="Q")
        assert "plan" in p.lower() and "connecteurs" in p.lower()
        s = llama_cerveau._PROMPT_STYLE.format(plan="PLAN", question="Q",
                                               contexte_court="CTX")
        assert "vocabulaire" in s.lower() and "connecteurs" in s.lower()

    def test_mode_riche_ne_touche_pas_les_factuels(self):
        ia = Aura1B()
        assert not ia._est_complexe("Quelle est la capitale du Japon ?")
        assert not ia._est_complexe("Calcule le carre de 4")

    def test_mode_riche_declenche_sur_ouvertes(self):
        ia = Aura1B()
        assert ia._est_complexe(
            "Redige une analyse comparee du solaire et de l'eolien en France")
        assert ia._est_complexe("Explique pourquoi le ciel est bleu et "
                                "donne ton avis argumente sur ce phenomene")
