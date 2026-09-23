"""Tests memoire de conversation : RLM (MIT), fenetre glissante, etat
evenementiel. L'historique complet vit dans un FICHIER consultable par
outil ; la fenetre du prompt ne contient que les derniers tours ; l'etat
civil est extrait par regex en JSON compact.
"""
import json

import pytest

from aura.memoire_conversation import MemoireConversation


@pytest.fixture()
def memoire(tmp_path):
    m = MemoireConversation(chemin=tmp_path / "conv.jsonl", fenetre=4)
    yield m
    # chaque fixture a son propre fichier : aucune fuite entre tests


class TestRlm:
    def test_historique_ecrit_en_fichier(self, memoire):
        memoire.ajouter("user", "je m'appelle Pierre")
        memoire.ajouter("assistant", "Enchante, Pierre.")
        lignes = memoire._chemin.read_text(encoding="utf-8").splitlines()
        assert len(lignes) == 2
        assert json.loads(lignes[0])["content"] == "je m'appelle Pierre"

    def test_recherche_trouve_le_vieux_tour(self, memoire):
        for i in range(30):
            memoire.ajouter("user", f"question numero {i} sur la physique")
            memoire.ajouter("assistant", f"reponse {i}")
        memoire.ajouter("user", "mon code secret est ZEBRE-42, oublie rien")
        for i in range(20):
            memoire.ajouter("user", f"autre fil {i}")
        hit = memoire.rechercher("ZEBRE")
        assert "ZEBRE-42" in hit

    def test_recherche_sans_resultat(self, memoire):
        memoire.ajouter("user", "bonjour")
        assert memoire.rechercher("xylophone-inexistant") == ""

    def test_recherche_insensible_accents_casse(self, memoire):
        memoire.ajouter("user", "J'habite a Caen depuis 2020")
        assert "Caen" in memoire.rechercher("caen")

    def test_outil_declenche_par_la_question(self, memoire):
        memoire.ajouter("user", "le mot de passe du wifi est CAROTTE-7")
        memoire.ajouter("assistant", "note.")
        for i in range(10):
            memoire.ajouter("user", f"blabla {i}")
            memoire.ajouter("assistant", f"ok {i}")
        r = memoire.tourner("cherche dans l'historique : CAROTTE")
        assert r is not None and "CAROTTE-7" in r

    def test_pas_doutil_sans_demande(self, memoire):
        assert memoire.tourner("quelle heure est-il ?") is None


class TestFenetreGlissante:
    def test_fenetre_tronquee(self, memoire):
        for i in range(10):
            memoire.ajouter("user", f"u{i}")
            memoire.ajouter("assistant", f"a{i}")
        f = memoire.fenetre()
        assert len(f) == 8                       # 4 tours = 8 messages
        assert f[-1]["content"] == "a9"          # le plus recent au bout
        assert all(t["content"] not in (f"u0", f"a0") for t in f[:2])

    def test_vider_keeps_fichier(self, memoire):
        memoire.ajouter("user", "secret : PIN-1234")
        memoire.vider()
        assert memoire.fenetre() == []
        assert memoire.rechercher("PIN-1234") != ""   # RLM preserve


class TestEtatEvenementiel:
    def test_nom_extrait(self, memoire):
        memoire.ajouter("user", "Je m'appelle Pierre")
        assert memoire.etat()["utilisateur"] == "Pierre"

    def test_possession_extraite(self, memoire):
        memoire.ajouter("user", "j'ai acheté 4 serveurs")
        assert memoire.etat()["serveurs"] == "4"

    def test_etat_json_compact(self, memoire):
        memoire.ajouter("user", "moi c'est Alice")
        memoire.ajouter("user", "j'ai 2 chatons")
        j = memoire.etat_json()
        assert json.loads(j) == {"utilisateur": "Alice", "chatons": "2"}
        assert len(j) < 80                        # quelques octets

    def test_dernier_etat_gagne(self, memoire):
        memoire.ajouter("user", "j'ai 4 serveurs")
        memoire.ajouter("user", "j'ai vendu mes serveurs, j'ai 1 serveur")
        assert memoire.etat()["serveur"] == "1"

    def test_bloc_systeme_injecte_etat(self, memoire):
        memoire.ajouter("user", "Je m'appelle Pierre")
        bloc = memoire.bloc_systeme("Tu es Aura.")
        assert "Tu es Aura." in bloc               # ancre preservee
        assert "Pierre" in bloc                    # etat ajoute
        assert "historique" in bloc                # outil annonce
