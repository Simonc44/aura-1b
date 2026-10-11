"""Tests de l'expert calcul de taux (taux x duree, total / duree, partage).

Defaut reel mesure par scripts/banc_100.py — 5 questions de la categorie
« raisonnement », toutes de la meme famille : le 1B ENONCE la formule au
lieu de calculer (« Pour trouver la distance parcourue par le train, on
peut utiliser la formule : ... »). Attendu : 240, 50, 120, 2, 60.
"""
import json

from aura import calcul_taux
from aura.orchestrateur import RontoGPT


class TestCalculTaux:
    def test_taux_fois_duree(self):
        # etait « on peut utiliser la formule : ... » : au banc, attendu 240
        assert calcul_taux.resoudre(
            "Un train roule a 80 km/h pendant 3 heures. Quelle distance "
            "parcourt-il ?") == "240"

    def test_total_sur_duree_vitesse(self):
        # au banc, attendu 50
        assert calcul_taux.resoudre(
            "Une voiture parcourt 150 km en 3 heures. Quelle est sa "
            "vitesse moyenne en km/h ?") == "50"

    def test_salaire_horaire(self):
        # au banc, attendu 120
        assert calcul_taux.resoudre(
            "Un ouvrier gagne 15 euros de l'heure. Il travaille 8 heures. "
            "Combien gagne-t-il ?") == "120"

    def test_partage_en_parts_egales(self):
        # au banc, attendu 2
        assert calcul_taux.resoudre(
            "Une pizza de 8 parts est partagee entre 4 personnes. Combien "
            "de parts chacune ?") == "2"

    def test_debit_fois_duree(self):
        # au banc, attendu 60
        assert calcul_taux.resoudre(
            "Un robinet remplit 12 litres par minute. Combien de litres en "
            "5 minutes ?") == "60"

    def test_autre_unite_de_temps(self):
        assert calcul_taux.resoudre(
            "Un tuyau debite 7 litres par seconde pendant 4 secondes, "
            "combien de litres ?") == "28"

    def test_abstention_si_unites_incompatibles(self):
        """Aucune conversion heures/minutes : on s'abstient, on ne devine pas.

        Repondre exigerait une conversion que le module ne fait pas — et une
        reponse fausse serait pire qu'une cascade vers le cerveau.
        """
        assert calcul_taux.resoudre(
            "Un robinet remplit 12 litres par minute. Combien de litres en "
            "2 heures ?") is None

    def test_abstention_sans_question(self):
        assert calcul_taux.resoudre(
            "Un train roule a 80 km/h pendant 3 heures.") is None

    def test_abstention_sans_taux_ni_partage(self):
        assert calcul_taux.resoudre(
            "Il y a 9 eleves et chacun a 4 cahiers, combien de cahiers ?"
        ) is None

    def test_le_taux_prix_exige_une_devise(self):
        """« 3 heures » n'est PAS un taux de 3 par heure.

        Sans devise obligatoire, « 150 km en 3 heures » se lisait comme un
        taux de 3/heure applique a 3 heures -> 9. Doit rester None.
        """
        assert calcul_taux.resoudre(
            "Une voiture parcourt 150 km en 3 heures. Quelle distance ?"
        ) is None

    def test_ninterfere_pas_avec_les_problemes_verbaux(self):
        """Les problemes racontees du banc ne doivent pas etre captures."""
        for q in (
            "Un cycliste roule 18 km puis 9 km, combien en tout ?",
            "Un gateau a 12 parts, on en coupe 5, combien reste-t-il ?",
            "Un bus contient 45 passagers. 17 descendent et 9 montent. "
            "Combien de passagers reste-t-il ?",
            "Un reservoir de 60 litres est rempli aux deux tiers. Combien "
            "de litres contient-il ?",
            "J'ai 15 euros et j'achete un livre a 7 euros, combien me "
            "reste-t-il ?",
        ):
            assert calcul_taux.resoudre(q) is None, q

    def test_branche_dans_le_pipeline(self, monkeypatch, tmp_path):
        """Le calcul de taux ne doit JAMAIS atteindre le cerveau.

        Le cache instantane du PROJET est isole ici, comme le fait
        scripts/banc_100.py via AURA_FILTRE_CACHE — sans quoi le score
        mesurerait la memoire du projet, plus le routage. Mesure :
        « train 80 km/h » est deja memorise dans le cache du projet, donc
        sans isolation c'est le cache qui repond AVANT les experts et le
        tier teste ne serait jamais exerce.
        """
        from aura import filtre_instantane as F, llama_cerveau
        monkeypatch.setattr(F, "_FICHIER", tmp_path / "cache_neuf.jsonl")
        monkeypatch.setattr(F, "_entrees", [])
        monkeypatch.setattr(F, "_vectoriseur", None)
        monkeypatch.setattr(F, "_matrice", None)
        monkeypatch.setattr(F, "_vecteurs", None)
        monkeypatch.setattr(
            llama_cerveau, "generer",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("le calcul de taux doit repondre")))
        r = RontoGPT().executer_detaille(
            "Un train roule a 80 km/h pendant 3 heures. Quelle distance "
            "parcourt-il ?")
        assert r["cerveau_choisi"] == "calcul-taux"
        assert r["reponse"] == "240"

    def test_ordre_calcul_verbal_prioritaire(self, monkeypatch):
        """Le tier verbal passe AVANT : il ne perd aucune de ses questions."""
        from aura import calcul_verbal as cv
        q = "J'ai 12 bonbons, j'en donne 5, combien m'en reste-t-il ?"
        assert cv.resoudre(q) == "7"       # verbal repond
        assert calcul_taux.resoudre(q) is None   # et le taux s'abstient


class TestEchantillonnageServeur:
    """Le chemin serveur (l'ACTIF) n'envoyait aucune penalite : boucles."""

    def _capture(self, monkeypatch) -> dict:
        from aura import serveur_lora as S
        pris: dict = {}

        class _Reponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "ok"}}]}).encode()

        def faux_urlopen(req, timeout=None):
            pris["corps"] = json.loads(req.data.decode())
            return _Reponse()

        monkeypatch.setattr(S, "_assurer", lambda: True)
        monkeypatch.setattr(S.urllib.request, "urlopen", faux_urlopen)
        assert S.chat([{"role": "user", "content": "bonjour"}], 16) == "ok"
        return pris["corps"]

    def test_penalites_envoyees(self, monkeypatch):
        # pas de AURA_SANS_DRY : c'est le cas par defaut qui est teste ici
        monkeypatch.delenv("AURA_SANS_DRY", raising=False)
        corps = self._capture(monkeypatch)
        assert corps["repeat_penalty"] == 1.1
        assert corps["min_p"] == 0.05
        assert corps["dry_multiplier"] == 0.8
        assert corps["dry_allowed_length"] == 2

    def test_dry_desactivable(self, monkeypatch):
        monkeypatch.setenv("AURA_SANS_DRY", "1")
        corps = self._capture(monkeypatch)
        assert "dry_multiplier" not in corps
        assert corps["repeat_penalty"] == 1.1
