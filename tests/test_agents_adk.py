"""Chaine ADK (opt-in AURA_ADK=1) : modele local simule, sans GGUF ni reseau."""

from aura import agents_adk
from aura import llama_cerveau


def test_inactif_par_defaut(monkeypatch):
    monkeypatch.delenv("AURA_ADK", raising=False)
    assert agents_adk.actif() is False
    assert agents_adk.planifier_adk("organise-moi un voyage à Rome") is None


def test_pas_de_plan_pour_une_question_simple(monkeypatch):
    monkeypatch.setenv("AURA_ADK", "1")
    assert agents_adk.planifier_adk("quelle heure est-il ?") is None


def test_plan_adk_accepte(monkeypatch):
    monkeypatch.setenv("AURA_ADK", "1")

    def faux_generer(question, systeme=None, historique=None, max_tokens=None, **kw):
        return "1. Choisir les dates\n2. Réserver le vol\n3. Préparer le budget"

    monkeypatch.setattr(llama_cerveau, "generer", faux_generer)
    plan = agents_adk.planifier_adk("organise-moi un voyage à Rome")
    assert plan is not None
    assert plan.splitlines()[0].startswith("1. Choisir les dates")
    assert len(plan.splitlines()) == 3


def test_plan_inexploitable_repli(monkeypatch):
    monkeypatch.setenv("AURA_ADK", "1")
    monkeypatch.setattr(llama_cerveau, "generer",
                        lambda *a, **k: "Bien sûr, voici mon avis.")
    assert agents_adk.planifier_adk("organise-moi un voyage à Rome") is None


def test_erreur_du_modele_repli(monkeypatch):
    monkeypatch.setenv("AURA_ADK", "1")

    def boom(*a, **k):
        raise RuntimeError("cerveau indisponible")

    monkeypatch.setattr(llama_cerveau, "generer", boom)
    assert agents_adk.planifier_adk("organise-moi un voyage à Rome") is None
