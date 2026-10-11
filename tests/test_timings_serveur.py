"""Tests des timings llama-server captures par `serveur_lora`.

llama-server renvoie, a chaque /v1/chat/completions, un bloc `timings`
(prompt_n, prompt_ms, predicted_n, predicted_ms...). RontoGPT le JETAIT : il ne
pouvait donc pas savoir combien de tokens il envoyait vraiment, ni ce que
coutait le prefill par rapport au decode. Ces tests verrouillent la capture.
"""
import json

import pytest

from aura import serveur_lora


class _Reponse:
    """Fausse reponse urllib : contexte + lecture (et iteration = flux SSE)."""

    def __init__(self, corps=None, lignes=None):
        self._corps = corps
        self._lignes = lignes or []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._corps).encode()

    def __iter__(self):
        return iter(self._lignes)


TIMINGS = {"prompt_n": 323, "prompt_ms": 1310.0, "prompt_per_second": 246.5,
           "predicted_n": 120, "predicted_ms": 6360.0,
           "predicted_per_second": 18.9}


@pytest.fixture(autouse=True)
def _timings_vides():
    """Chaque test part d'un etat propre (module global persistant)."""
    serveur_lora._oublier_timings()
    yield
    serveur_lora._oublier_timings()


class TestMemoriserTimings:
    def test_cle_inconnue_ignoree(self):
        serveur_lora._memoriser_timings(
            {"timings": {**TIMINGS, "bidule": 3, "x": "texte"}})
        assert serveur_lora.derniers_timings() == TIMINGS

    def test_pas_de_timings_laisse_vide(self):
        serveur_lora._memoriser_timings({"choices": [{}]})
        assert serveur_lora.derniers_timings() == {}

    def test_timings_non_dict_ignore(self):
        serveur_lora._memoriser_timings({"timings": "aucun"})
        assert serveur_lora.derniers_timings() == {}

    def test_valeur_non_numerique_ignoree(self):
        serveur_lora._memoriser_timings(
            {"timings": {"prompt_n": 12, "prompt_ms": None}})
        assert serveur_lora.derniers_timings() == {"prompt_n": 12.0}

    def test_derniers_timings_est_une_copie(self):
        serveur_lora._memoriser_timings({"timings": TIMINGS})
        copie = serveur_lora.derniers_timings()
        copie["prompt_n"] = 99999
        assert serveur_lora.derniers_timings()["prompt_n"] == 323.0

    def test_oublier_vide(self):
        serveur_lora._memoriser_timings({"timings": TIMINGS})
        serveur_lora._oublier_timings()
        assert serveur_lora.derniers_timings() == {}


class TestCaptureDansLesAppels:
    def test_poster_capture(self, monkeypatch):
        corps = {"choices": [{"message": {"content": "  salut  "}}],
                 "timings": TIMINGS}
        monkeypatch.setattr(serveur_lora.urllib.request, "urlopen",
                            lambda req, timeout: _Reponse(corps))
        assert serveur_lora._poster({"messages": []}) == "salut"
        assert serveur_lora.derniers_timings()["prompt_n"] == 323.0
        assert serveur_lora.derniers_timings()["predicted_per_second"] == 18.9

    def test_poster_sans_timings_ne_garde_pas_les_precedents(
            self, monkeypatch):
        """Un appel qui ne fournit pas de timings ne doit pas laisser
        croire qu'il a coute le prix de l'appel d'avant."""
        serveur_lora._memoriser_timings({"timings": TIMINGS})
        monkeypatch.setattr(
            serveur_lora.urllib.request, "urlopen",
            lambda req, timeout: _Reponse(
                {"choices": [{"message": {"content": "ok"}}]}))
        serveur_lora._poster({"messages": []})
        assert serveur_lora.derniers_timings() == {}

    def test_flux_capture_le_dernier_evenement(self, monkeypatch):
        lignes = [
            b'data: {"choices": [{"delta": {"content": "bon"}}]}\n',
            b"\n",
            b'data: {"choices": [{"delta": {"content": "jour"}}]}\n',
            b'data: {"choices": [{"delta": {}}], "timings": '
            + json.dumps(TIMINGS).encode() + b"}\n",
            b"data: [DONE]\n",
        ]
        monkeypatch.setattr(serveur_lora.urllib.request, "urlopen",
                            lambda req, timeout: _Reponse(lignes=lignes))
        morceaux: list[str] = []
        assert serveur_lora._chat_stream(
            {"messages": []}, morceaux.append) == "bonjour"
        assert morceaux == ["bon", "jour"]
        assert serveur_lora.derniers_timings()["predicted_n"] == 120.0

    def test_statut_expose_les_timings(self, monkeypatch):
        monkeypatch.setattr(serveur_lora, "_actifs", lambda: False)
        monkeypatch.setattr(serveur_lora, "_existe_serveur", lambda: False)
        serveur_lora._memoriser_timings({"timings": TIMINGS})
        assert serveur_lora.statut()["timings"]["prompt_n"] == 323.0
