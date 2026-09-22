"""Tests des 5 ameliorations : multi-LoRA, dynamic compute, RLSS, triplets, verrou."""
import json

import pytest

from aura import agents, rlss, serveur_lora, verrou


class TestMixMultiLora:
    def test_activer_simple_renvoie_mixte(self, monkeypatch):
        appels = {}

        def faux_mixte(poids):
            appels["p"] = poids
            return True

        monkeypatch.setattr(serveur_lora, "activer_mixte", faux_mixte)
        assert serveur_lora.activer("math") is True
        assert appels["p"] == {"math": 1.0}

    def test_activer_mixte_rejette_serveur_mort(self, monkeypatch):
        monkeypatch.setattr(serveur_lora, "_assurer", lambda: False)
        assert serveur_lora.activer_mixte({"math": 0.7}) is False

    def test_activer_mixte_cache_meme_mix(self, monkeypatch):
        serveur_lora._actif = (("math", 0.7),)   # deja applique
        try:
            assert serveur_lora.activer_mixte({"math": 0.7}) is True  # sans reseau
        finally:
            serveur_lora._actif = None

    def test_corps_envoye_avec_tous_les_ids(self, monkeypatch):
        envoye = {}

        def faux_post(req, timeout):
            envoye["corps"] = json.loads(req.data.decode())
            import io
            return io.BytesIO(b'{"success": true}')

        monkeypatch.setattr(serveur_lora, "_assurer", lambda: True)
        monkeypatch.setattr("urllib.request.urlopen", faux_post)
        assert serveur_lora.activer_mixte({"math": 0.7, "web": 0.3}) is True
        echelles = {c["id"]: c["scale"] for c in envoye["corps"]}
        assert echelles == {0: 0.7, 1: 0.3, 2: 0.0, 3: 0.0}


class TestDynamicCompute:
    def test_generer_complexe_passe_le_flag(self, monkeypatch):
        capturé = {}

        def faux_chat(messages, max_tokens=256):
            capturé["messages"] = messages
            capturé["budget"] = max_tokens
            return "La reponse."

        monkeypatch.setattr(serveur_lora, "_actifs", lambda: True)
        monkeypatch.setattr(serveur_lora, "_actif", (("math", 1.0),))
        monkeypatch.setattr(serveur_lora, "chat", faux_chat)
        from aura import llama_cerveau
        r = llama_cerveau.generer("pourquoi le ciel est bleu ?",
                                  complexe=True, categorie="general")
        assert r == "La reponse."
        # le prompt de reflexion masquee est present, budget double
        textes = " ".join(m["content"] for m in capturé["messages"])
        assert "<thinking>" in textes
        assert capturé["budget"] >= 400

    def test_separer_reflexion_masque_le_brouillon(self):
        from aura.llama_cerveau import separer_reflexion
        propre, reflexion = separer_reflexion(
            "<thinking>analyse interne</thinking>Reponse finale.")
        assert propre == "Reponse finale."
        assert reflexion == "analyse interne"


class TestRlss:
    def test_graphe_vide_annule(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rlss, "_GRAPHE", tmp_path / "absent.jsonl")
        bilan = rlss.session(nb=5)
        assert bilan["exercices"] == 0

    def test_valider_tolerant(self):
        assert rlss._valider("Paris est la capitale", "Paris")
        assert rlss._valider("La ville est ROME.", "Rome")
        assert not rlss._valider("Je ne sais pas.", "Rome")

    def test_exercice_deterministe(self):
        ex = rlss._fabriquer_exercice({"sujet": "la France", "objet": "Paris"})
        assert ex is not None
        categorie, question, attendu = ex
        assert categorie in ("general", "web")
        assert "la France" in question
        assert attendu == "Paris"

    def test_session_ecrit_les_succes(self, monkeypatch, tmp_path):
        graphe = tmp_path / "graphe.jsonl"
        graphe.write_text(json.dumps({"s": "capitale de la France",
                                      "r": "fait", "o": "Paris"}) + "\n",
                          encoding="utf-8")
        monkeypatch.setattr(rlss, "_GRAPHE", graphe)
        sortie = tmp_path / "rlss.jsonl"
        monkeypatch.setattr(rlss, "_SORTIE", sortie)
        monkeypatch.setattr(rlss, "_CORRECTIONS", tmp_path / "corrections.jsonl")
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda q, **k: "C'est Paris, la capitale.")
        bilan = rlss.session(nb=1)
        assert bilan["succes"] == 1 and bilan["echecs"] == 0
        exemple = json.loads(sortie.read_text(encoding="utf-8").splitlines()[0])
        assert exemple["user"] and "Paris" in exemple["assistant"]


class TestTriplets:
    def test_phrase_capitale_devient_triplet(self):
        texte = ("Paris est la capitale de la France depuis des siecles. "
                 "La ville est situee dans le nord du pays, sur la Seine. "
                 "Elle compte environ deux millions d'habitants aujourd'hui. "
                 "Le tourisme y est tres developpe chaque annee.")
        r = agents.en_triplets(texte, "capitale de la France")
        assert "capitale_de" in r

    def test_texte_sans_relation_intact(self):
        texte = "Le chat dort sur le canape. Il fait beau aujourd'hui."
        assert agents.en_triplets(texte, "meteo") == texte

    def test_texte_court_intact(self):
        assert agents.en_triplets("Court.", "q") == "Court."


class TestVerrou:
    def test_reforge_refusee_sans_cle_privee(self, monkeypatch, tmp_path):
        monkeypatch.setattr(verrou, "_CLE_PRIVEE", tmp_path / "absente.pem")
        assert verrou.autoriser_reforge() is False

    def test_rapport_complet(self, monkeypatch, tmp_path):
        monkeypatch.setattr(verrou, "_CLE_PRIVEE", tmp_path / "cle.pem")
        monkeypatch.setattr(verrou, "_CLE_PUBLIQUE", tmp_path / "pub.pem")
        r = verrou.rapport()
        assert r["reforge_autorisee"] is False
        assert "zones_donnees" in r

    def test_chaine_sans_signature_faux(self, monkeypatch, tmp_path):
        monkeypatch.setattr(verrou, "_RACINE", tmp_path)
        assert verrou.verifier_chaine() is False
