"""Tests du harnais d'evaluation personnel (scripts/harnais_evaluation.py).

Le harnais mesure le SYSTEME (pipeline / paquet scelle / exe). On teste
ici sa LOGIQUE (verification, score, dataset) — pas les cibles lourdes :
pipeline/aef/exe sont valides en reel (voir rapports eval_*.json).
"""
import importlib

import pytest

harnais = importlib.import_module("scripts.harnais_evaluation")


class TestVerifier:
    def test_numerique_trouve_le_bon_nombre(self):
        assert harnais.verifier("il vous reste 6 euros au total", "6",
                                "numerique")

    def test_numerique_refuse_mauvais_nombre(self):
        assert not harnais.verifier("il vous reste 10 euros", "6",
                                    "numerique")

    def test_numerique_ignores_les_autres_nombres(self):
        # « j'ai 10 euros » ne doit pas faire passer « attendu 6 »
        assert not harnais.verifier("j'ai 10 euros", "6", "numerique")

    def test_contient_insensible_casse_accents(self):
        assert harnais.verifier("Vous êtes Frédérique !", "frederique",
                                "contient")

    def test_exact_normalise(self):
        assert harnais.verifier("  42  ", "42", "exact")

    def test_aucun_compte_pas_dans_le_score(self):
        # mode "aucun" : tour de construction, toujours vrai
        assert harnais.verifier("n'importe quoi", "x", "aucun")

    def test_mode_inconnu_refuse(self):
        with pytest.raises(ValueError):
            harnais.verifier("reponse", "x", "mode-bizarre")


class TestDataset:
    def test_dataset_integre_coherent(self):
        # chaque item compte doit avoir attendu + mode valide
        for d in harnais.DATASET:
            assert d["question"]
            if d.get("mode", "aucun") != "aucun":
                assert d.get("attendu"), d
                assert d["mode"] in ("numerique", "contient", "exact")

    def test_dataset_externe_jsonl(self, tmp_path):
        f = tmp_path / "extra.jsonl"
        f.write_text(
            '{"question": "combien font 2 plus 2", "attendu": "4", '
            '"mode": "numerique", "categorie": "math"}\n'
            '{"question": "tour de construction", "mode": "aucun", '
            '"categorie": "memoire"}\n', encoding="utf-8")
        items = harnais.charger_dataset(str(f))
        assert len(items) == 2
        assert items[0]["categorie"] == "math"

    def test_dataset_vide_refuse(self, tmp_path):
        f = tmp_path / "vide.jsonl"
        f.write_text("\n \n", encoding="utf-8")
        with pytest.raises(SystemExit):
            harnais.charger_dataset(str(f))


class TestEvaluer:
    def test_score_et_categories(self):
        dataset = [
            {"question": "q1", "attendu": "1", "mode": "numerique",
             "categorie": "math"},
            {"question": "q2", "attendu": "2", "mode": "numerique",
             "categorie": "math"},
            {"question": "q3", "attendu": "Aura", "mode": "contient",
             "categorie": "identite"},
        ]
        reponses = {"q1": "resultat 1", "q2": "resultat 3",   # 1 faux
                    "q3": "Je suis Aura"}

        def poser(q):
            return reponses[q]

        r = harnais.evaluer(poser, lambda: None, dataset)
        assert r["passes"] == 2 and r["total"] == 3
        assert r["score"] == pytest.approx(2 / 3)
        assert r["categories"]["math"]["passe"] == 1
        assert r["categories"]["math"]["echecs"][0]["attendu"] == "2"
        assert r["categories"]["identite"]["passe"] == 1

    def test_reset_entre_sequences(self):
        # deux blocs de sequence distincte : reinitialiser doit etre appele
        # entre chaque bloc (la conversation memoire est isolee)
        appels = []
        dataset = [
            {"sequence": 1, "question": "a", "mode": "aucun",
             "categorie": "memoire"},
            {"sequence": 1, "question": "b", "attendu": "b",
             "mode": "contient", "categorie": "memoire"},
            {"sequence": 2, "question": "c", "mode": "aucun",
             "categorie": "memoire"},
        ]
        harnais.evaluer(lambda q: q, lambda: appels.append(1), dataset)
        assert len(appels) == 2          # debut seq 1 + debut seq 2

    def test_aucun_ne_compte_pas(self):
        dataset = [
            {"question": "construire", "mode": "aucun", "categorie": "m"},
            {"question": "verifier", "attendu": "v", "mode": "contient",
             "categorie": "m"},
        ]
        r = harnais.evaluer(lambda q: "reponse v", lambda: None, dataset)
        assert r["total"] == 1 and r["passes"] == 1
