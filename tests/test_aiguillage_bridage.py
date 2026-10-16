"""Aiguillage neuro-symbolique : les 4 bugs racines du 12 oct. 2026.

1. RAG bride : 3 morceaux (classique) / 10 (complexe), jamais 154.
2. Calcul deterministe : (4583 * 921) + 14582 = 4235525, contre-verifie.
3. Code verrouille sur la sandbox locale (potcode), JAMAIS sur le web.
4. Enigmes de logique envoyees au mini-SAT : le 1B n'extrait que le formalisme.

Tout est hors reseau : le web et le cerveau sont simules.
"""
from fractions import Fraction

import pytest

from aura import (agents, apis_publiques, embeddings, llama_cerveau, logique,
                  memoire_web, potcode, registre_experts)
from aura import filtre_instantane as f0
from aura.orchestrateur import RontoGPT
from aura.routeur import RouteurIntelligent


# ── 1. Bridage du RAG ───────────────────────────────────────────────────

def _faux_web(n=154):
    return "\n".join(
        f"Titre : page {i} | Extrait : python rust comparaison fonction "
        f"tri {i} | Source : https://exemple.org/{i}" for i in range(n))


class TestBridageRag:
    def test_limites_par_defaut_et_complexe(self):
        assert memoire_web.limite_morceaux("qui a invente le tri rapide") == 3
        assert memoire_web.limite_morceaux(
            "Compare en detail Python et Rust") == 10
        assert memoire_web.limite_morceaux("x " * 200) == 10   # tres long
        assert memoire_web.RAG_MAX_ABSOLU == 10

    @pytest.mark.parametrize("valeur, attendu", [
        ("999", 10), ("0", 1), ("abc", 3), ("5", 5), ("154", 10)])
    def test_variable_d_environnement_bornee(self, valeur, attendu):
        assert memoire_web._entier_borne(
            valeur, 3, 1, memoire_web.RAG_MAX_ABSOLU) == attendu

    def test_limiter_morceaux_154_vers_3(self):
        ctx = _faux_web()
        sortie = memoire_web.limiter_morceaux(ctx, question="une question")
        assert len(sortie.splitlines()) == 3

    def test_limiter_morceaux_plafond_absolu(self):
        sortie = memoire_web.limiter_morceaux(_faux_web(), maximum=500)
        assert len(sortie.splitlines()) == 10

    def test_chercher_iteratif_classique_3_morceaux(self, monkeypatch):
        monkeypatch.setattr(embeddings, "encoder", lambda t: None)
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda requete, max_resultats=3, timeout=None:
                            _faux_web())
        sortie = memoire_web.chercher_iteratif(
            "fonction tri python", max_resultats=50)
        assert len(sortie.splitlines()) <= 3

    def test_chercher_iteratif_complexe_10_au_plus(self, monkeypatch):
        monkeypatch.setattr(embeddings, "encoder", lambda t: None)
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda requete, max_resultats=3, timeout=None:
                            _faux_web())
        sortie = memoire_web.chercher_iteratif(
            "Compare en detail python rust fonction tri", max_resultats=500)
        n = len(sortie.splitlines())
        assert 3 < n <= 10

    def test_compresseur_borne_meme_un_contexte_court(self):
        ctx = "\n".join(f"python fonction {i}" for i in range(8))
        assert len(agents.compresser(ctx, "python fonction").splitlines()) <= 3

    def test_verrou_final_dans_le_prompt(self):
        messages: list[dict] = []
        faits = "\n".join(f"- fait {i} : est x{i}" for i in range(154))
        llama_cerveau._ajouter_contextes(messages, _faux_web(), "", faits)
        web = messages[0]["content"]
        assert web.count("Titre :") <= 10
        assert messages[2]["content"].count("- fait") <= 10


# ── 2. Calcul deterministe ──────────────────────────────────────────────

class TestCalculDeterministe:
    @pytest.mark.parametrize("question", [
        "Combien font (4583 * 921) + 14582",
        "Combien font (4583 * 921) + 14582 ?",
        "combien font (4583 x 921) + 14582",
        "(4583*921)+14582",
        "4583*921+14582",
    ])
    def test_cas_du_bug(self, question):
        rep = f0.repondre(question)
        assert rep is not None
        assert rep.replace(" ", "") == "4235525"

    @pytest.mark.parametrize("question, attendu", [
        ("2+3*4", "14"), ("(2+3)*4", "20"), ("combien font 2 + 3 fois 4", "14"),
        ("100-25-5", "70"), ("2 puissance 3 plus 1", "9"),
    ])
    def test_priorites(self, question, attendu):
        assert f0.repondre(question).replace(" ", "") == attendu

    def test_les_deux_evaluateurs_s_accordent(self):
        for expr in ("(4583 * 921) + 14582", "-2**2", "2**3**2", "7 // 2",
                     "1/3 + 1/6", "(1+2)*(3+4)-5"):
            assert float(f0._evaluer_independant(expr)) == pytest.approx(
                float(f0._calculer(expr)))

    def test_desaccord_des_evaluateurs_force_l_abstention(self, monkeypatch):
        monkeypatch.setattr(f0, "_evaluer_independant",
                            lambda expr: Fraction(1887823))
        assert f0._calcul_direct("Combien font (4583 * 921) + 14582") is None

    def test_x_n_est_pas_toujours_une_multiplication(self):
        assert f0.repondre("2x+3") is None

    def test_le_cache_ne_resservira_plus_un_calcul_faux(self, monkeypatch,
                                                        tmp_path):
        fichier = tmp_path / "cache.jsonl"
        fichier.write_text(
            '{"q": "combien font (4583 * 921) + 14582", "r": "1887823"}\n'
            '{"q": "capitale de la france", "r": "Paris."}\n',
            encoding="utf-8")
        monkeypatch.setattr(f0, "_FICHIER", fichier)
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])
        monkeypatch.setattr(f0, "_vecteurs", None)
        f0._charger_cache()
        assert [e["r"] for e in f0._entrees] == ["Paris."]
        assert "1887823" not in fichier.read_text(encoding="utf-8")

    def test_un_calcul_n_est_jamais_memorise(self, monkeypatch, tmp_path):
        monkeypatch.setattr(f0, "_FICHIER", tmp_path / "cache.jsonl")
        monkeypatch.setattr(f0, "_vectoriseur", None)
        monkeypatch.setattr(f0, "_matrice", None)
        monkeypatch.setattr(f0, "_entrees", [])
        monkeypatch.setattr(f0, "_vecteurs", None)
        f0.enregistrer("Combien font (4583 * 921) + 14582", "1887823")
        assert f0.stats()["entrees_cache"] == 0

    def test_pipeline_complet(self):
        r = RontoGPT().executer_detaille("Combien font (4583 * 921) + 14582")
        assert r["cerveau_choisi"] == "niveau0-instantane"
        assert r["reponse"].replace(" ", "") == "4235525"


# ── 3. Aiguillage du code ───────────────────────────────────────────────

CODE_OUI = [
    "Ecris une fonction Python qui trie une liste",
    "écris une fonction",
    "donne moi un code python pour inverser une chaine",
    "def somme(a, b): return a + b  -- corrige-le",
    "assert resoudre([1]) == [1]  fais passer ce test",
    "Comment inverser une liste en Python ?",
    "Crée un script bash pour sauvegarder un dossier",
    "write a function that sorts a list",
    "Code une classe Pile",
    "ecris une fonction python qui renvoie la version de python en 2026",
]
CODE_NON = [
    "Quelle est la dernière version de Python ?",
    "Qu'est-ce que Python ?",
    "Quel est le code postal de Nantes ?",
    "Qui a écrit le code civil ?",
    "Quel est le programme TV ce soir ?",
    "qui est le président de la France",
    "Quand est sorti Java 21 ?",
    "donne moi le code promo de ce site",
    "Combien font 2+2",
]


class TestAiguillageCode:
    @pytest.mark.parametrize("question", CODE_OUI)
    def test_demande_de_code_reconnue(self, question):
        assert potcode.est_demande_code(question)
        assert RouteurIntelligent.aiguiller(question) == "code"

    @pytest.mark.parametrize("question", CODE_NON)
    def test_question_ordinaire_pas_verrouillee(self, question):
        assert not potcode.est_demande_code(question)

    def test_contrat_sandbox(self):
        assert potcode.demande_sandbox_python(
            "ecris une fonction python qui trie une liste")
        assert not potcode.demande_sandbox_python(
            "ecris une fonction javascript qui trie un tableau")
        assert not potcode.demande_sandbox_python(
            "ecris une fonction python qui lit un fichier")
        assert not potcode.demande_sandbox_python("Crée un script bash")

    def test_code_python_route_vers_la_sandbox(self, monkeypatch):
        monkeypatch.setattr(RontoGPT, "_resoudre_par_code",
                            lambda self, q, mode_auto=False: "def resoudre(): pass")
        r = RontoGPT()._expert_special(
            "Ecris une fonction Python avec des assert qui renvoie le carre",
            None, None)
        assert r["cerveau_choisi"] == "potcode-1b+sandbox"

    def test_code_hors_contrat_ne_passe_ni_par_logique_ni_par_web(
            self, monkeypatch):
        appels = []
        monkeypatch.setattr(RontoGPT, "_resoudre_par_logique",
                            lambda self, q, t0=None: appels.append("logique"))
        monkeypatch.setattr(RontoGPT, "_resoudre_par_code",
                            lambda self, q, mode_auto=False: appels.append("code"))
        assert RontoGPT()._expert_special(
            "Ecris une fonction JavaScript qui trie un tableau",
            None, None) is None
        assert appels == []

    def test_le_rag_ne_detourne_pas_une_demande_de_code(self):
        ctx = registre_experts.Contexte(
            question="Ecris une fonction Python qui trie une liste",
            est_puzzle=False, a_donnees=False)
        assert RontoGPT._expert_applicable("rag-binaire", ctx) is False

    @pytest.mark.parametrize("question", [
        "Ecris une fonction Python qui renvoie le carre d'un nombre",
        # mot-cle temporel + code : partait sur DuckDuckGo avant le correctif
        "Ecris une fonction JavaScript qui trie un tableau, version 2026",
        # forme de definition + code
        "Qu'est-ce que le tri fusion ? Ecris la fonction en Python",
    ])
    def test_pipeline_sans_aucune_recherche_web(self, monkeypatch, question):
        def interdit(*a, **k):
            raise AssertionError("le web ne doit JAMAIS etre appele pour du code")

        monkeypatch.setattr(memoire_web, "chercher_iteratif", interdit)
        monkeypatch.setattr(memoire_web, "chercher", interdit)
        monkeypatch.setattr(apis_publiques, "chercher", interdit)
        monkeypatch.setattr(f0, "enregistrer", lambda q, r: None)
        bloc = ("```python\ndef resoudre(x):\n    return x * x\n"
                "assert resoudre(3) == 9\nassert resoudre(0) == 0\n"
                "assert resoudre(-2) == 4\n```")
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda q, *a, **kw: bloc)
        r = RontoGPT().executer_detaille(question)
        assert "web" not in r["experts"]
        assert r["contexte_web"] == ""


# ── 4. Enigmes de logique ───────────────────────────────────────────────

ENIGME = ("Pierre, Jeanne et Loïc occupent trois postes différents : "
          "directeur, comptable et technicien. Pierre n'est pas directeur. "
          "Jeanne n'est pas comptable. Loïc n'est pas technicien. Jeanne n'est "
          "pas directrice. Qui occupe quel poste ?")
FORMALISME = ("ENTITES : Pierre, Jeanne, Loic\n"
              "DOMAINE : directeur, comptable, technicien\n"
              "TOUS DIFFERENTS\n"
              "CONDITION : Pierre != directeur\n"
              "CONDITION : Jeanne != comptable\n"
              "CONDITION : Loic != technicien\n"
              "CONDITION : Jeanne != directeur")


class TestAiguillageLogique:
    def test_enigme_d_attribution_detectee(self):
        assert logique.est_enigme(ENIGME)
        assert RouteurIntelligent.aiguiller(ENIGME) == "logique"
        assert logique.est_enigme(
            "Anna, Bob et Cal portent chacun une couleur différente : rouge, "
            "vert ou bleu. Anna ne porte pas le vert. Bob ne porte pas le rouge.")
        assert logique.est_enigme("Paul dit que Marie est menteuse. Qui ment ?")

    @pytest.mark.parametrize("question", [
        "Qui est le président de la France ?",
        "Quel est le métier de Pierre et Jeanne ?",
        "Combien font 12 + 15 + 14 ?",
        "Marie, Paul et Luc ont 12, 14 et 16 ans. Qui est le plus âgé ?",
        "Ecris une histoire avec Pierre, Jeanne et Loïc qui ne se parlent pas",
    ])
    def test_pas_de_faux_positif(self, question):
        assert not logique.est_enigme(question)

    def test_le_solveur_tranche_l_enigme(self):
        puzzle = logique.parser_puzzle(FORMALISME)
        r = logique.resoudre(puzzle)
        assert r["statut"] == "unique"
        s = r["solutions"][0]
        assert (s["Pierre"], s["Jeanne"], s["Loic"]) == (
            "comptable", "technicien", "directeur")
        assert logique.entites_dans_question(puzzle, ENIGME)

    def test_le_1b_ne_sert_qu_a_extraire(self, monkeypatch):
        appels = []

        def faux_generer(q, *a, **kw):
            appels.append(kw.get("systeme"))
            return FORMALISME

        monkeypatch.setattr(llama_cerveau, "generer", faux_generer)
        rep = RontoGPT()._resoudre_par_logique(ENIGME)
        assert "Pierre : comptable" in rep
        assert "Jeanne : technicien" in rep
        assert "Loic : directeur" in rep
        assert appels and all(s == logique._SYSTEME_LOGIQUE for s in appels)

    def test_entites_recopiees_d_un_exemple_rejetees(self, monkeypatch):
        monkeypatch.setattr(
            llama_cerveau, "generer",
            lambda q, *a, **kw: "ENTITES : Alice, Bob, Cal\n"
                            "DOMAINE : rouge, vert, bleu\nTOUS DIFFERENTS\n"
                            "CONDITION : Alice == rouge\n"
                            "CONDITION : Bob != vert")
        assert RontoGPT()._resoudre_par_logique(ENIGME) is None

    def test_extraction_impossible_refus_honnete_sans_cache(self, monkeypatch):
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda q, *a, **kw: "Je pense que la reponse est oui")

        def pas_de_cache(q, r):
            raise AssertionError("un refus ne se memorise pas")

        monkeypatch.setattr(f0, "enregistrer", pas_de_cache)
        r = RontoGPT()._expert_special(ENIGME, None, None)
        assert r["cerveau_choisi"] == "logique-non-formalisee"
        assert "formaliser" in r["reponse"]

    def test_pipeline_sans_web_et_sans_consigne_brute(self, monkeypatch):
        def interdit(*a, **k):
            raise AssertionError("pas de web pour une enigme de logique")

        monkeypatch.setattr(memoire_web, "chercher_iteratif", interdit)
        monkeypatch.setattr(memoire_web, "chercher", interdit)
        monkeypatch.setattr(apis_publiques, "chercher", interdit)
        monkeypatch.setattr(f0, "enregistrer", lambda q, r: None)
        monkeypatch.setattr(
            llama_cerveau, "generer",
            lambda q, *a, **kw: (FORMALISME
                             if kw.get("systeme") == logique._SYSTEME_LOGIQUE
                             else "BEGAIEMENT DU 1B"))
        r = RontoGPT().executer_detaille(ENIGME)
        assert "Pierre : comptable" in r["reponse"]
        assert "BEGAIEMENT" not in r["reponse"]
