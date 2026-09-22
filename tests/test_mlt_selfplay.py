"""Tests MLT autonome + self-play (genere, resout, juge, reintegre)."""
import json

import pytest

from aura import graphe_faits, selfplay


# ── MLT : force par triplet, renforcement, recherche ponderee ─────────

@pytest.fixture()
def graphe_tmp(monkeypatch, tmp_path):
    """Redirige le graphe vers un fichier temporaire avec 2 triplets."""
    fichier = tmp_path / "graphe.jsonl"
    donnees = [
        {"s": "Paris", "r": "capitale de", "o": "France", "f": 1.0,
         "d": 0},
        {"s": "Tokyo", "r": "capitale de", "o": "Japon", "f": 1.0,
         "d": 0},
    ]
    fichier.write_text("\n".join(json.dumps(d) for d in donnees) + "\n",
                       encoding="utf-8")
    monkeypatch.setattr(graphe_faits, "_FICHIER", str(fichier))
    monkeypatch.setattr(graphe_faits, "_triplets", [])
    monkeypatch.setattr(graphe_faits, "_index", {})
    monkeypatch.setattr(graphe_faits, "_charg\u00e9", False)
    return fichier


class TestMLT:
    def test_fichier_vide_ok(self, graphe_tmp):
        stats = graphe_faits.stats()
        assert stats["triplets"] == 2

    def test_renforcement_augmente_la_force(self, graphe_tmp):
        assert graphe_faits.renforcer("Paris", "France") is True
        assert graphe_faits.renforcer("Paris", "France") is True
        # 1.0 initial + 2 renforcements = 3.0 (fichier reecrit a l'etat final)
        contenu = graphe_tmp.read_text(encoding="utf-8")
        triplets = [json.loads(l) for l in contenu.strip().splitlines()]
        paris = next(t for t in triplets if t["s"] == "Paris")
        assert paris["f"] == 3.0

    def test_renforcement_inconnu_cree_le_lien(self, graphe_tmp):
        # lien inconnu = schema de pensee eprouve : memorise au vol
        assert graphe_faits.renforcer("Inconnu", "Nullepart") is True
        contenu = graphe_tmp.read_text(encoding="utf-8")
        assert "associe_a" in contenu

    def test_recherche_ponderee_par_la_force(self, graphe_tmp):
        # Paris renforce x3 -> doit sortir seul (Tokyo reste sous le seuil)
        for _ in range(3):
            graphe_faits.renforcer("Paris", "France")
        resultats = graphe_faits.chercher("capitale de la France")
        assert "Paris" in resultats
        assert "Tokyo" not in resultats

    def test_trois_champs_renforce_sur_nouveau_triplet(self, graphe_tmp):
        # un triplet ajoute SANS force puis renforce doit survivre
        graphe_faits.ajouter("Rome", "capitale de", "Italie", source="test")
        assert graphe_faits.renforcer("Rome", "Italie") is True


# ── Self-play : generateur, juge, boucle fermee ────────────────────────

class TestGenerateurDefis:
    def test_generer_enigme_a_une_solution_exacte(self):
        d = selfplay.generer_enigme(nb_personnes=3)
        assert d is not None
        assert d["type"] == "logique"
        assert d["attendu"] in selfplay._LIEUX
        assert d["cible"] in d["question"]

    def test_generer_defi_calcul_reponse_exacte(self):
        d = selfplay.generer_defi_calcul()
        assert d["type"] == "calcul"
        assert d["attendu"].isdigit()

    def test_generer_defi_niveau1_toujours_calcul(self, monkeypatch):
        monkeypatch.setattr(selfplay.random, "random", lambda: 0.9)
        d = selfplay.generer_defi(niveau=2)
        assert d["type"] == "calcul"


class TestJuge:
    def test_valide(self):
        assert selfplay._juger("Bibliotheque", "bibliotheque") is True

    def test_rejete_si_absent(self):
        assert selfplay._juger("Je ne sais pas", "bibliotheque") is False

    def test_rejete_si_vide(self):
        assert selfplay._juger("", "bibliotheque") is False

    def test_rejete_si_trop_long(self):
        assert selfplay._juger("x" * 300 + " bibliotheque",
                               "bibliotheque") is False


class TestSession:
    def test_session_avec_cerveau_fictif(self, monkeypatch, tmp_path):
        # cerveau toujours parfait + graphe temporaire + sortie temporaire
        monkeypatch.setattr(selfplay, "_SORTIE", tmp_path / "sp.jsonl")
        monkeypatch.setattr(selfplay, "_faits_graphe", lambda n=8: [])
        _cerveau_parfait(monkeypatch, "42")

        bilan = selfplay.session(nb_defis=4, niveau=1)
        assert bilan["defis"] == 4
        assert bilan["succes"] + bilan["echecs"] == 4
        # le taux ne peut pas depasser 1
        assert 0.0 <= bilan["taux"] <= 1.0

    def test_session_reintegre_les_succes(self, monkeypatch, tmp_path):
        sortie = tmp_path / "sp.jsonl"
        monkeypatch.setattr(selfplay, "_SORTIE", sortie)
        monkeypatch.setattr(selfplay, "_faits_graphe", lambda n=8: [])
        _cerveau_parfait(monkeypatch, "42")
        monkeypatch.setattr(selfplay, "_juger", lambda r, a: True)

        selfplay.session(nb_defis=3, niveau=1)
        lignes = sortie.read_text(encoding="utf-8").strip().splitlines()
        assert len(lignes) == 3
        for ligne in lignes:
            exemple = json.loads(ligne)
            assert exemple["type"] == "calcul"
            assert exemple["assistant"] == "42"

    def test_session_echec_enregistre_correction(self, monkeypatch, tmp_path):
        monkeypatch.setattr(selfplay, "_SORTIE", tmp_path / "sp.jsonl")
        monkeypatch.setattr(selfplay, "_faits_graphe", lambda n=8: [])
        from aura import llama_cerveau, autoamelioration
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "je ne sais pas")
        corrections = []
        monkeypatch.setattr(autoamelioration,
                            "enregistrer_correction",
                            lambda q, rf, c, cat="": corrections.append(c))

        bilan = selfplay.session(nb_defis=2, niveau=1)
        assert bilan["echecs"] == 2
        assert len(corrections) == 2

    def test_session_renforce_le_graphe(self, monkeypatch, tmp_path):
        monkeypatch.setattr(selfplay, "_SORTIE", tmp_path / "sp.jsonl")
        faits = [{"s": "Paris", "r": "capitale de", "o": "France"}]
        monkeypatch.setattr(selfplay, "_faits_graphe", lambda n=8: faits)
        _cerveau_parfait(monkeypatch, "une reponse")
        monkeypatch.setattr(selfplay, "_juger", lambda r, a: True)
        renforts = []
        monkeypatch.setattr(graphe_faits, "renforcer",
                            lambda s, o, delta=1.0:
                            renforts.append((s, o)) or True)
        # defi logique force -> le succes doit renforcer le graphe
        defi = {"type": "logique", "question": "q", "attendu": "jardin",
                "faits": {}, "cible": "X"}
        monkeypatch.setattr(selfplay, "generer_defi", lambda niveau=1: defi)

        selfplay.session(nb_defis=1, niveau=2)
        assert renforts == [("Paris", "France")]


def _cerveau_parfait(monkeypatch, reponse: str = "42") -> None:
    """Mock le module llama_cerveau (session() l'importe localement)."""
    from aura import llama_cerveau
    monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: reponse)
