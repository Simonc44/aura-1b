"""API publiques (liste public-apis) : detection, formatage, cache, wiring.

Aucun test n'ouvre de vraie connexion : `_json` est remplace ou urllib est
mocke ; le conftest coupe AURA_APIS pour toute la suite (le module est
re-active ici via monkeypatch).
"""
import pytest

from aura import apis_publiques


@pytest.fixture(autouse=True)
def apis_activees(monkeypatch):
    """Re-active la brique (le conftest la coupe) + vide les caches."""
    monkeypatch.setenv("AURA_APIS", "1")
    apis_publiques._cache.clear()
    apis_publiques._echecs.clear()
    yield


# ── detection : rien ne doit changer hors sujet ──────────────────────────

class TestDetection:
    HORS_SUJET = [
        # questions du banc de QUALITE (toutes doivent rester intactes)
        "Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que Clara. "
        "Clara a 4 ans. Quel age a Alice ?",
        "Quel est le prix du carburant en France aujourd'hui ?",
        "Combien font 6 x 7 ?",
        "Qu'est-ce que la photosynthese ?",
        "J'ai 3 pommes, j'en mange 1, combien il m'en reste ?",
        "Si j'ai 10 euros et que je dépense 4 euros, combien me reste-t-il ?",
        "Redige une analyse comparee du solaire et de l'eolien en France",
        # questions hors API
        "Qui es-tu ?",
        "Explique-moi le fonctionnement d'un ordinateur quantique",
    ]

    @pytest.mark.parametrize("question", HORS_SUJET)
    def test_hors_sujet_non_detecte(self, question):
        """Zero octet envoye : aucune regex ne doit correspondre."""
        assert not apis_publiques.detecte(question)

    @pytest.mark.parametrize("question, connecteur", [
        ("Quel temps fait-il à Paris ?", "meteo"),
        ("Météo de Lyon demain ?", "meteo"),
        ("Quelle est la température à Nice ?", "meteo"),
        ("Combien vaut 100 dollars en euros ?", "devises"),
        ("100 € en dollars ?", "devises"),
        ("Quel est le taux de change USD en EUR ?", "devises"),
        ("Définition de pragmatique ?", "wiktionary"),
        ("Que signifie aberrant ?", "wiktionary"),
        ("Qui a écrit Le Petit Prince ?", "openlibrary"),
        ("Quel livre parle de la guerre de 14-18 ?", "openlibrary"),
        ("Où se trouve la tour Eiffel ?", "nominatim"),
        ("Où est Tokyo ?", "nominatim"),
    ])
    def test_detecte_le_bon_connecteur(self, question, connecteur):
        assert apis_publiques.detecte(question)
        assert any(
            c.nom == connecteur
            and any(m.search(question) for m in c.motifs)
            for c in apis_publiques._CONNECTEURS)

    def test_desactive_par_env(self, monkeypatch):
        monkeypatch.setenv("AURA_APIS", "0")
        assert not apis_publiques.detecte("Quel temps fait-il à Paris ?")
        assert apis_publiques.chercher("Quel temps fait-il à Paris ?") == ""


# ── formatage : _json remplace par des reponses reelles capturees ────────

class TestFormatage:
    def test_meteo_formate(self, monkeypatch):
        vus: list[str] = []

        def _faux_json(url: str):
            vus.append(url)
            if "geocoding" in url:
                return {"results": [{"name": "Paris",
                                     "latitude": 48.85341,
                                     "longitude": 2.3488,
                                     "country": "France"}]}
            return {"current": {"time": "2026-10-03T14:00",
                                "temperature_2m": 15.34,
                                "apparent_temperature": 14.81,
                                "weather_code": 0,
                                "wind_speed_10m": 12.4}}

        monkeypatch.setattr(apis_publiques, "_json", _faux_json)
        texte = apis_publiques.chercher("Quel temps fait-il à Paris ?")
        assert len(vus) == 2                      # geocodage + previsions
        assert "Paris, France" in texte
        assert "15,3 °C" in texte and "14,8 °C" in texte
        assert "ciel dégagé" in texte
        assert "12 km/h" in texte and "14:00" in texte

    def test_meteo_sans_ville_fait_zero_appel(self, monkeypatch):
        def _boom(url: str):
            raise AssertionError("aucun appel réseau sans ville")

        monkeypatch.setattr(apis_publiques, "_json", _boom)
        assert apis_publiques.chercher("Quel temps fait-il ?") == ""

    def test_devise_convertit(self, monkeypatch):
        def _faux_json(url: str):
            assert "frankfurter" in url
            return {"amount": 100.0, "base": "USD",
                    "date": "2026-10-02", "rates": {"EUR": 89.09}}

        monkeypatch.setattr(apis_publiques, "_json", _faux_json)
        texte = apis_publiques.chercher("Combien vaut 100 dollars en euros ?")
        assert "100 USD = 89,09 EUR" in texte
        assert "2026-10-02" in texte

    def test_devise_meme_devise_ignoree(self, monkeypatch):
        monkeypatch.setattr(
            apis_publiques, "_json",
            lambda url: pytest.fail("euro -> euro ne doit pas appeler"))
        assert apis_publiques._devises("Combien valent 10 euros en euros ?") == ""

    def test_definition_extrait_la_section(self, monkeypatch):
        # extrait Wiktionary REAL (endpoint prop=extracts, mode plaintext)
        extrait = (
            "\n== Français ==\n\n\n=== Étymologie ===\n"
            "Du latin pragmaticus (« relatif aux affaires »).\n\n\n"
            "=== Adjectif ===\n\npragmatique \\pʁaɡ.ma.tik\\\n\n"
            "Qualifie le choix d'une décision, d'une action, orienté vers "
            "l'efficacité pratique plutôt que vers la théorie.\n\n"
            "=== Dérivés ===\npragmatiquement.\n\n")

        def _faux_json(url: str):
            assert "wiktionary" in url
            return {"query": {"pages": {"148080": {
                "pageid": 148080, "title": "pragmatique",
                "extract": extrait}}}}

        monkeypatch.setattr(apis_publiques, "_json", _faux_json)
        texte = apis_publiques.chercher("Définition de pragmatique ?")
        assert texte.startswith("Définition de « pragmatique » :")
        assert "Qualifie le choix" in texte       # pas d'etymologie dedans
        assert "\\" not in texte                  # prononciation sautee

    def test_definition_mot_absent(self, monkeypatch):
        def _faux_json(url: str):
            return {"query": {"pages": {"1": {"title": "zzzz",
                                              "missing": ""}}}}

        monkeypatch.setattr(apis_publiques, "_json", _faux_json)
        assert apis_publiques._definitions("Que signifie zzzz ?") == ""

    def test_livre_identifie(self, monkeypatch):
        def _faux_json(url: str):
            assert "openlibrary" in url
            return {"docs": [{"title": "Le petit prince",
                              "author_name": ["Antoine de Saint-Exupéry"],
                              "first_publish_year": 1943}]}

        monkeypatch.setattr(apis_publiques, "_json", _faux_json)
        texte = apis_publiques.chercher("Qui a écrit Le Petit Prince ?")
        assert "« Le petit prince »" in texte
        assert "Antoine de Saint-Exupéry" in texte
        assert "1943" in texte

    def test_geo_formate(self, monkeypatch):
        def _faux_json(url: str):
            assert "nominatim" in url
            return [{"display_name": "Tokyo, Japon",
                     "lat": "35.6768601", "lon": "139.7638947"}]

        monkeypatch.setattr(apis_publiques, "_json", _faux_json)
        texte = apis_publiques.chercher("Où est Tokyo ?")
        assert "Tokyo, Japon" in texte
        assert "35,6769" in texte and "139,764" in texte


# ── reseau : cache TTL et memoire d'echec (urllib mocke) ─────────────────

class _Reponse:
    def __init__(self, corps: bytes):
        self._corps = corps

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, taille: int = -1) -> bytes:
        return self._corps


class TestReseau:
    def test_cache_un_seul_appel_http(self, monkeypatch):
        import json as _json_mod
        appels: list[str] = []

        def _faux_urlopen(requete, timeout=None):
            appels.append(requete.full_url)
            return _Reponse(_json_mod.dumps({"docs": []}).encode())

        monkeypatch.setattr("urllib.request.urlopen", _faux_urlopen)
        url = "https://openlibrary.org/search.json?q=x&limit=1"
        assert apis_publiques._json(url) == {"docs": []}
        assert apis_publiques._json(url) == {"docs": []}
        assert apis_publiques._json(url) == {"docs": []}
        assert len(appels) == 1                   # 2e/3e appel : cache

    def test_echec_reseau_memoise(self, monkeypatch):
        appels: list[int] = []

        def _erreur(requete, timeout=None):
            appels.append(1)
            raise OSError("reseau off")

        monkeypatch.setattr("urllib.request.urlopen", _erreur)
        url = "https://api.frankfurter.app/latest?amount=1"
        assert apis_publiques._json(url) is None
        assert apis_publiques._json(url) is None
        assert len(appels) == 1                   # echec = 1 essai / 120 s

    def test_desactivee_sans_appel(self, monkeypatch):
        def _erreur(requete, timeout=None):
            raise AssertionError("AURA_APIS=0 ne doit rien appeler")

        monkeypatch.setattr("urllib.request.urlopen", _erreur)
        monkeypatch.setenv("AURA_APIS", "0")
        assert apis_publiques._json("https://x/y") is None


# ── wiring orchestrateur : la tache part en parallele du web ─────────────

class TestOrchestrateur:
    @staticmethod
    def _ia(tmp_path):
        from aura.memoire_conversation import MemoireConversation
        from aura.orchestrateur import Aura1B
        ia = Aura1B()
        ia._historique = MemoireConversation(chemin=tmp_path / "c.jsonl")
        return ia

    @staticmethod
    def _isole_pipeline(monkeypatch):
        """Chemin normal : ni cache ni expert qui intercepte, pas de LLM.

        Le fan-out reel tourne (avec les API mockees par chaque test) ;
        toute generation inattendue leve une assertion claire.
        """
        from aura import (agents, calcul_verbal, fast_lang,
                          filtre_instantane, llama_cerveau, memoire_web,
                          rag_binaire)
        from aura.orchestrateur import Aura1B
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(filtre_instantane, "enregistrer", lambda q, r: None)
        monkeypatch.setattr(rag_binaire, "chercher", lambda q: None)
        monkeypatch.setattr(fast_lang.moteur, "repondre", lambda q: None)
        monkeypatch.setattr(calcul_verbal, "resoudre", lambda q: None)
        monkeypatch.setattr(memoire_web, "chercher", lambda q, **k: "")
        monkeypatch.setattr(memoire_web, "chercher_enrichi",
                            lambda q, **k: "")
        monkeypatch.setattr(agents, "planifier", lambda q, **k: None)
        monkeypatch.setattr(Aura1B, "_expert_special",
                            staticmethod(lambda *a, **k: None))
        monkeypatch.setattr(Aura1B, "_a_besoin_verification",
                            staticmethod(lambda *a, **k: False))

    # ── fan-out : la tache API part en parallele et est mise de cote ────

    def test_api_mise_de_cote_par_le_fan_out(self, monkeypatch):
        from aura import apis_publiques as ap
        monkeypatch.setattr(ap, "detecte", lambda q: True)
        monkeypatch.setattr(ap, "chercher", lambda q: "METEO API")
        from aura.orchestrateur import Aura1B
        ia = Aura1B()
        ctx, formule = ia._executer_experts(
            set(), "Quel temps fait-il à Paris ?", None, None, False)
        assert (ctx, formule) == ("", "")         # web absent
        assert ia._contexte_api == "METEO API"     # mais stashee

    def test_api_se_merge_au_web(self, monkeypatch):
        from aura import apis_publiques as ap, memoire_web
        from aura.orchestrateur import Aura1B
        monkeypatch.setattr(ap, "detecte", lambda q: True)
        monkeypatch.setattr(ap, "chercher", lambda q: "METEO API")
        monkeypatch.setattr(memoire_web, "chercher",
                            lambda q, **k: "FAIT WEB")
        ia = Aura1B()
        ctx, formule = ia._executer_experts(
            {"web"}, "meteo Paris", None, None, False)
        assert "FAIT WEB" in ctx                   # web rendu tel quel
        assert "METEO API" not in ctx              # API NON compressee ici
        assert ia._contexte_api == "METEO API"
        assert formule == ""

    def test_detecte_fausse_sans_tache(self, monkeypatch):
        from aura import apis_publiques as ap
        from aura.orchestrateur import Aura1B
        monkeypatch.setattr(ap, "detecte", lambda q: False)
        ia = Aura1B()
        ctx, formule = ia._executer_experts(
            set(), "nimporte quoi", None, None, False)
        assert (ctx, formule) == ("", "")         # chemin rapide inchange
        assert ia._contexte_api == ""

    # ── pipeline : la donnee API est livree / survit au compresseur ─────

    def test_pipeline_api_seule_livree_sans_generation(self, monkeypatch,
                                                       tmp_path):
        """API seule -> reponse = la donnee structuree, ZERO appel LLM.

        Contre-mesure observee en direct : avec generation, le 1B refusait
        « les informations meteo en temps reel » alors que la meteo etait
        dans le contexte.
        """
        from aura import apis_publiques, filtre_instantane, llama_cerveau
        from aura.orchestrateur import Aura1B
        self._isole_pipeline(monkeypatch)
        monkeypatch.setattr(apis_publiques, "detecte", lambda q: True)
        donnee = "Météo à Paris, France (14:00) : 14,7 °C, ciel dégagé."
        monkeypatch.setattr(apis_publiques, "chercher", lambda q: donnee)
        enregistres: list = []
        monkeypatch.setattr(filtre_instantane, "enregistrer",
                            lambda q, r: enregistres.append(r))

        def _boom(*args, **kwargs):
            raise AssertionError("generation inattendue")

        monkeypatch.setattr(Aura1B, "_generer", _boom)
        monkeypatch.setattr(llama_cerveau, "generer", _boom)

        det = self._ia(tmp_path).executer_detaille(
            "Quel temps fait-il à Paris ?")
        assert det["cerveau_choisi"] == "extraction-api"
        assert det["reponse"] == donnee
        assert enregistres == [donnee]

    def test_pipeline_api_survit_au_compresseur(self, monkeypatch, tmp_path):
        """API + web long : la ligne de donnees passe APRES le top-3."""
        from aura import agents, llama_cerveau
        from aura.orchestrateur import Aura1B
        self._isole_pipeline(monkeypatch)
        captures: list = []

        def _faux_experts(self, experts, question, X, y, riche):
            self._contexte_api = "API LIGNE DE DONNEES."
            phrases = " ".join(
                f"Phrase numero {i} avec des mots utiles et un chiffre 7."
                for i in range(1, 8))            # ~380 car. > seuil 320
            return phrases, ""

        monkeypatch.setattr(Aura1B, "_executer_experts", _faux_experts)

        def _faux_generer(self, *args, **kwargs):
            captures.append(args)
            return "REPONSE FINALE"

        monkeypatch.setattr(Aura1B, "_generer", _faux_generer)
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("generation inattendue")))

        det = self._ia(tmp_path).executer_detaille(
            "Quel temps fait-il à Paris ?")
        assert det["reponse"] == "REPONSE FINALE"
        assert captures, "la generation aurait du etre appelee"
        contexte = captures[-1][1]                # (question, ctx, formule)
        assert "API LIGNE DE DONNEES." in contexte
        assert len(contexte) < 300                # web compresse au top-3
        assert "Phrase numero 6" not in contexte  # les phrases sont coupees
