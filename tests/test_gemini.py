"""Repli Gemini (API Google, HTTP direct) : aucun appel reseau, urllib mocke.

Deux familles de tests :
- le JUGEMENT (`raison_du_repli`) : quand une reponse locale doit partir ;
- l'INTEGRATION (`executer_detaille`) : Gemini remplace la reponse locale
  faible, ne rejuge jamais un tier exact, et laisse la reponse locale intacte
  quand il se tait.
"""
import io
import json
import logging
import urllib.error

import pytest

from aura import gemini

_VARIABLES = ("AURA_GEMINI", "AURA_GEMINI_DOMAINES", "AURA_GEMINI_REGLES",
              "AURA_GEMINI_HISTORIQUE", "AURA_GEMINI_MODELE",
              "AURA_GEMINI_TIMEOUT", "AURA_GEMINI_MAX_TOKENS",
              "AURA_GEMINI_REFLEXION")


@pytest.fixture(autouse=True)
def _isole(monkeypatch, tmp_path):
    """Pas de cle, pas de `.env` reel : le jugement part d'un etat connu.

    La fixture `_gemini_coupe` du conftest coupe la brique dans toute la
    suite ; ici on la re-active (comme les tests d'apis_publiques le font
    pour AURA_APIS) et on neutralise le `.env` de la racine en pointant
    `_RACINE` sur un dossier temporaire vide.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for v in _VARIABLES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(gemini, "_RACINE", tmp_path)
    monkeypatch.setenv("AURA_GEMINI", "1")
    gemini.reinitialiser()
    yield
    gemini.reinitialiser()


@pytest.fixture
def avec_cle(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fausse-cle-test")


def _res(reponse, cerveau="gemma3-1b", **extra):
    return {"reponse": reponse, "cerveau_choisi": cerveau, **extra}


def _longue(n):
    """n phrases DISTINCTES (pas de boucle), ~45 caracteres chacune."""
    return " ".join(f"Phrase numero {i} sur la liberte et ses limites."
                    for i in range(n))


# -- activation ----------------------------------------------------------------

def test_sans_cle_jamais_de_repli():
    assert gemini.configure() is False
    assert gemini.raison_du_repli("q", _res("")) is None
    assert gemini.repondre("bonjour") is None


def test_interrupteur_general(avec_cle, monkeypatch):
    assert gemini.configure() is True
    monkeypatch.setenv("AURA_GEMINI", "0")
    assert gemini.configure() is False
    assert gemini.raison_du_repli("q", _res("")) is None
    assert gemini.repondre("bonjour") is None


def test_cle_lue_dans_le_env(tmp_path):
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=cle-env\n", encoding="utf-8")
    assert gemini.configure() is True
    assert gemini._cle() == "cle-env"


def test_la_cle_de_l_environnement_prime_sur_le_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=cle-env\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "cle-shell")
    assert gemini._cle() == "cle-shell"


# -- le jugement ---------------------------------------------------------------

Q_NEUTRE = "Donne-moi des idees de repas pour la semaine"
REP_NEUTRE = ("Varie les legumes, les feculents et les proteines, et prepare "
              "quelques bases a l'avance pour gagner du temps.")
Q_PROBLEME = ("Lea a 12 billes et en perd 5 puis elle en gagne 8, combien en "
              "a-t-elle a la fin de la partie ?")
Q_FACTUEL = "Qui est le fondateur de Microsoft ?"


@pytest.mark.parametrize("question,reponse,attendu", [
    (Q_NEUTRE, "", "vide"),
    (Q_NEUTRE, "   ", "vide"),
    (Q_NEUTRE, "[RontoGPT] Reflexion interrompue", "erreur"),
    (Q_NEUTRE, "Je suis désolé, mais je ne peux pas répondre.", "refus"),
    (Q_NEUTRE, "Cette information m'échappe : je ne sais pas.", "aveu"),
    ("Décris la mer",
     " ".join(["La mer est grande et bleue et calme aujourd'hui."] * 5),
     "boucle"),
    ("Quelle est la différence entre un virus et une bactérie ?",
     "Quelle est la différence entre un virus et une bactérie ?", "echo"),
    ("Explique pourquoi le ciel est bleu", "À cause de la lumière.", "court"),
    ("Rédige une dissertation sur la liberté", _longue(6), "redaction"),
    (Q_PROBLEME, "Elle en a 15.", "probleme"),
    (Q_FACTUEL, "Bill Gates.", "factuel"),
])
def test_reponses_faibles_partent(avec_cle, question, reponse, attendu):
    assert gemini.raison_du_repli(question, _res(reponse)) == attendu


@pytest.mark.parametrize("question,reponse", [
    (Q_NEUTRE, REP_NEUTRE),
    ("Explique pourquoi le ciel est bleu", _longue(6)),
    # une reponse factuelle AVEC une source n'est plus une devinette
    (Q_FACTUEL, "Selon la source, Bill Gates."),
])
def test_reponses_correctes_restent(avec_cle, question, reponse):
    r = _res(reponse, contexte_web="extrait verifie")
    assert gemini.raison_du_repli(question, r) is None


def test_une_reponse_factuelle_sans_source_part(avec_cle):
    # meme reponse, mais AUCUN extrait web : c'est une devinette du 1B
    assert gemini.raison_du_repli(Q_FACTUEL, _res("Bill Gates.")) == "factuel"
    assert gemini.raison_du_repli(
        Q_FACTUEL, _res("Bill Gates.", contexte_web="extrait")) is None


def test_un_probleme_de_code_ne_part_pas_en_probleme(avec_cle):
    q = ("Ecris une fonction python qui renvoie la somme de 12 et 5 puis "
         "la multiplie par 3 et affiche le resultat final")
    assert gemini.raison_du_repli(q, _res("def f(): pass")) is None


def test_une_formule_exacte_ne_part_pas_en_probleme(avec_cle):
    assert gemini.raison_du_repli(
        Q_PROBLEME, _res("Elle en a 15.", formule="12 - 5 + 8")) is None


@pytest.mark.parametrize("cerveau", [
    "niveau0-instantane", "wikipedia", "pot-1b+ast", "constante",
    "constante-capacite", "horloge", "calcul-verbal", "calcul-taux",
    "rag-binaire", "micro-faits", "cache-semantique", "extraction-api",
    "clarification", "titulaires", "local", "",
])
def test_un_tier_exact_n_est_jamais_rejuge(avec_cle, cerveau):
    # reponse qui avouerait : sur un tier exact elle reste telle quelle
    assert gemini.raison_du_repli(
        Q_NEUTRE, _res("Je ne sais pas.", cerveau)) is None


@pytest.mark.parametrize("cerveau", ["gemma3-1b", "llama-1b", "def-web+1b",
                                     "extraction-web", "qwen-1.5b"])
def test_un_tier_generatif_est_rejugeable(avec_cle, cerveau):
    assert gemini.raison_du_repli(
        Q_NEUTRE, _res("Je ne sais pas.", cerveau)) == "aveu"


@pytest.mark.parametrize("cerveau", ["niveau0-instantane", "wikipedia", "local"])
def test_une_reponse_vide_part_quel_que_soit_le_tier(avec_cle, cerveau):
    assert gemini.raison_du_repli(Q_NEUTRE, _res("", cerveau)) == "vide"


@pytest.mark.parametrize("prefixe", ["[Logique] Je ne sais pas.",
                                     "[Code] Je ne sais pas."])
def test_une_reponse_verifiee_n_est_pas_rejugee(avec_cle, prefixe):
    assert gemini.raison_du_repli(Q_NEUTRE, _res(prefixe)) is None


def test_une_relance_de_clarification_n_est_pas_une_reponse_faible(avec_cle):
    r = _res("Je ne sais pas.", "gemma3-1b", en_attente=True)
    assert gemini.raison_du_repli(Q_NEUTRE, r) is None


def test_mauvaise_langue(avec_cle):
    question = ("Explique pourquoi le ciel est bleu et donne les details de "
                "ce phenomene physique")
    anglais = (
        "the sky is blue because the light of the sun is scattered by the "
        "molecules of the air and the blue light of the spectrum is "
        "dispersed more than the red light which travels straight through "
        "the atmosphere without any deviation and this is what you see when "
        "you look at the sky")
    assert gemini.raison_du_repli(question, _res(anglais)) == "langue"


def test_domaine_designe_faible(avec_cle, monkeypatch):
    monkeypatch.setenv("AURA_GEMINI_DOMAINES", "code, math")
    assert gemini.raison_du_repli(Q_NEUTRE, _res(REP_NEUTRE), "math") == "domaine"
    assert gemini.raison_du_repli(Q_NEUTRE, _res(REP_NEUTRE), "web") is None
    # un domaine faible ne reecrit pas non plus un tier exact
    assert gemini.raison_du_repli(
        Q_NEUTRE, _res(REP_NEUTRE, "calcul-verbal"), "math") is None


def test_regles_selectionnees(avec_cle, monkeypatch):
    monkeypatch.setenv("AURA_GEMINI_REGLES", "vide")
    assert gemini.raison_du_repli(Q_NEUTRE, _res("")) == "vide"
    assert gemini.raison_du_repli(Q_NEUTRE, _res("Je ne sais pas.")) is None
    # une regle inconnue ne reactive rien
    monkeypatch.setenv("AURA_GEMINI_REGLES", "nimporte")
    assert gemini.raison_du_repli(Q_NEUTRE, _res("")) is None


# -- l'appel -------------------------------------------------------------------

def _faux_urlopen(vus, texte="  Santiago.  "):
    def faux(req, timeout):
        vus["url"] = req.full_url
        vus["cle"] = req.get_header("X-goog-api-key")
        vus["corps"] = json.loads(req.data.decode("utf-8"))
        vus["timeout"] = timeout
        corps = json.dumps({"candidates": [{"content": {"parts": [{"text": texte}]}}]})
        return io.BytesIO(corps.encode("utf-8"))
    return faux


def test_appel_mocke_renvoie_le_texte(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    q = "Quelle est la capitale du Chili ?"
    assert gemini.repondre(q) == "Santiago."
    assert vus["url"] == (f"{gemini.URL_BASE}/{gemini.MODELE_DEFAUT}"
                          ":generateContent")
    assert vus["cle"] == "fausse-cle-test"
    assert vus["timeout"] == 45
    corps = vus["corps"]
    assert corps["generationConfig"]["maxOutputTokens"] == 4096
    assert corps["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}
    assert corps["systemInstruction"]["parts"][0]["text"]
    assert corps["contents"][-1] == {"role": "user", "parts": [{"text": q}]}


def test_modele_et_budget_configurables(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setenv("AURA_GEMINI_MODELE", "gemini-2.0-flash")
    monkeypatch.setenv("AURA_GEMINI_MAX_TOKENS", "512")
    monkeypatch.setenv("AURA_GEMINI_TIMEOUT", "5")
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    gemini.repondre("Quelle est la capitale du Chili ?")
    assert "gemini-2.0-flash:generateContent" in vus["url"]
    assert vus["corps"]["generationConfig"]["maxOutputTokens"] == 512
    # la reflexion n'existe que sur la famille 2.5
    assert "thinkingConfig" not in vus["corps"]["generationConfig"]
    assert vus["timeout"] == 5


def test_valeur_numerique_invalide_retombe_sur_le_defaut(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setenv("AURA_GEMINI_TIMEOUT", "beaucoup")
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    gemini.repondre("Quelle est la capitale du Chili ?")
    assert vus["timeout"] == 45


def test_la_reflexion_n_est_jamais_livree(avec_cle, monkeypatch):
    corps = json.dumps({"candidates": [{"content": {"parts": [
        {"text": "je reflechis", "thought": True}, {"text": "Santiago."}]}}]})
    monkeypatch.setattr(gemini.urllib.request, "urlopen",
                        lambda req, timeout: io.BytesIO(corps.encode("utf-8")))
    assert gemini.repondre("Quelle est la capitale du Chili ?") == "Santiago."


@pytest.mark.parametrize("donnees", [
    {"candidates": []},
    {"candidates": [{"content": {"parts": []}}]},
    {"candidates": [{"content": {"parts": [{"text": "   "}]},
                     "finishReason": "MAX_TOKENS"}]},
    {"promptFeedback": {"blockReason": "SAFETY"}},
])
def test_reponse_sans_texte_renvoie_none(avec_cle, monkeypatch, donnees):
    corps = json.dumps(donnees)
    monkeypatch.setattr(gemini.urllib.request, "urlopen",
                        lambda req, timeout: io.BytesIO(corps.encode("utf-8")))
    assert gemini.repondre("Quelle est la capitale du Chili ?") is None
    assert gemini.derniere_erreur()


def test_extraits_web_joints_et_bornes(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setattr(gemini.urllib.request, "urlopen", _faux_urlopen(vus))
    q = "Quelle est la capitale du Chili ?"
    gemini.repondre(q, contexte_web="Santiago : capitale. " + "x" * 5000)
    contenu = vus["corps"]["contents"][-1]["parts"][0]["text"]
    assert contenu.startswith(q)
    assert "EXTRAITS WEB" in contenu and "Santiago : capitale." in contenu
    assert len(contenu) < len(q) + 3200


def test_l_historique_ne_part_que_pour_une_question_contextuelle(avec_cle):
    hist = [{"role": "user", "content": "je m'appelle Simon"},
            {"role": "assistant", "content": "Enchante Simon"}]
    long_q = "Donne-moi trois idees de repas equilibres pour toute la semaine"
    contenus = gemini._contenus(long_q, "", hist)
    assert [c["role"] for c in contenus] == ["user"]


def test_l_historique_part_pour_une_relance(avec_cle, monkeypatch):
    monkeypatch.setenv("AURA_GEMINI_HISTORIQUE", "2")
    hist = [{"role": "user", "content": "je parle de Paris"},
            {"role": "assistant", "content": "Paris est une ville."}]
    contenus = gemini._contenus("et sa population ?", "", hist)
    assert [c["role"] for c in contenus] == ["user", "model", "user"]
    assert contenus[0]["parts"][0]["text"] == "je parle de Paris"


def test_erreur_reseau_non_bloquante_et_cle_absente_des_logs(
        avec_cle, monkeypatch, caplog):
    def boom(req, timeout):
        raise OSError("x-goog-api-key: fausse-cle-test refuse")

    monkeypatch.setattr(gemini.urllib.request, "urlopen", boom)
    with caplog.at_level(logging.WARNING, logger="aura.gemini"):
        assert gemini.repondre("Quelle est la capitale du Chili ?") is None
    assert gemini.derniere_erreur()
    assert "fausse-cle-test" not in caplog.text


def test_cle_refusee_met_le_repli_en_pause(avec_cle, monkeypatch):
    def boom(req, timeout):
        corps = json.dumps({"error": {"message": "API key not valid"}}).encode()
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(corps))

    monkeypatch.setattr(gemini.urllib.request, "urlopen", boom)
    assert gemini.repondre("Quelle est la capitale du Chili ?") is None
    assert "cle API refusee" in gemini.derniere_erreur()
    # la pause empeche de rappeler Google a chaque question
    assert gemini.raison_du_repli(Q_NEUTRE, _res("")) is None


# -- integration dans executer_detaille ---------------------------------------

def _pipeline_factice(monkeypatch, reponse_locale, cerveau="gemma3-1b",
                      contexte_web="extrait web"):
    """RontoGPT sans charger le modele : seul le repli est reel."""
    from aura import orchestrateur, cache_reponses
    from aura.orchestrateur import RontoGPT

    ia = RontoGPT.__new__(RontoGPT)
    monkeypatch.setattr(cache_reponses, "cle_reponse", lambda q, X, y: None)
    monkeypatch.setattr(cache_reponses, "memoriser", lambda c, r: None)
    monkeypatch.setattr(
        RontoGPT, "_executer_detaille",
        lambda self, q, X=None, y=None, t0=None:
        {"reponse": reponse_locale, "cerveau_choisi": cerveau,
         "contexte_web": contexte_web})
    monkeypatch.setattr(orchestrateur, "_BUDGET_S", 0)
    ia._derniere_categorie = "general"
    return ia


def test_pipeline_repli_gemini_si_local_vide(avec_cle, monkeypatch):
    vus = {}
    monkeypatch.setattr(gemini, "repondre",
                        lambda q, **kw: vus.update(kw) or "reponse Gemini")
    ia = _pipeline_factice(monkeypatch, "")
    r = ia.executer_detaille("question de test")
    assert r["reponse"] == "reponse Gemini"
    assert r["cerveau_choisi"] == "gemini"
    assert r["repli"] == {"fournisseur": "gemini", "raison": "vide",
                          "reponse_locale": ""}
    assert vus["contexte_web"] == "extrait web"


def test_pipeline_garde_la_reponse_locale_valide(avec_cle, monkeypatch):
    appels = []
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: appels.append(q) or "X")
    ia = _pipeline_factice(monkeypatch, "Paris est la capitale de la France.")
    r = ia.executer_detaille("question de test")
    assert r["reponse"] == "Paris est la capitale de la France."
    assert appels == []
    assert "repli" not in r


def test_pipeline_ne_rejuge_pas_un_tier_exact(avec_cle, monkeypatch):
    appels = []
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: appels.append(q) or "X")
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.", "niveau0-instantane")
    r = ia.executer_detaille("Quelle est la capitale du Chili ?")
    assert r["reponse"] == "Je ne sais pas."
    assert appels == []


def test_pipeline_echec_gemini_laisse_la_reponse_locale(avec_cle, monkeypatch):
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: None)
    monkeypatch.setattr(gemini, "derniere_erreur", lambda: "quota depasse")
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.")
    r = ia.executer_detaille("Quelle est la capitale du Chili ?")
    assert r["reponse"] == "Je ne sais pas."
    assert r["cerveau_choisi"] == "gemma3-1b"
    assert r["repli_echec"] == "quota depasse"


def test_pipeline_remplace_la_mauvaise_reponse_du_cache(avec_cle, monkeypatch):
    from aura import filtre_instantane
    vus = []
    monkeypatch.setattr(filtre_instantane, "mettre_a_jour",
                        lambda q, r: vus.append((q, r)) or True)
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: "Santiago.")
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.")
    ia.executer_detaille("Quelle est la capitale du Chili ?")
    assert vus == [("Quelle est la capitale du Chili ?", "Santiago.")]


def test_pipeline_remplace_la_reponse_dans_la_fenetre(
        avec_cle, monkeypatch, tmp_path):
    from aura.memoire_conversation import MemoireConversation
    monkeypatch.setattr(gemini, "repondre", lambda q, **kw: "Santiago.")
    ia = _pipeline_factice(monkeypatch, "Je ne sais pas.")
    ia._historique = MemoireConversation(chemin=tmp_path / "h.jsonl")
    ia._historique.ajouter("user", "Quelle est la capitale du Chili ?")
    ia._historique.ajouter("assistant", "Je ne sais pas.")
    ia.executer_detaille("Quelle est la capitale du Chili ?")
    assert ia._historique.fenetre()[-1] == {"role": "assistant",
                                            "content": "Santiago."}


def test_pipeline_envoie_les_tours_precedents(avec_cle, monkeypatch, tmp_path):
    from aura.memoire_conversation import MemoireConversation
    vus = {}
    monkeypatch.setattr(gemini, "repondre",
                        lambda q, **kw: vus.update(kw) or "Santiago.")
    ia = _pipeline_factice(monkeypatch, "")
    ia._historique = MemoireConversation(chemin=tmp_path / "h.jsonl")
    ia._historique.ajouter("user", "je parle du Chili")
    ia._historique.ajouter("assistant", "Le Chili est un pays.")
    ia._historique.ajouter("user", "et sa capitale ?")      # reponse locale vide
    ia.executer_detaille("et sa capitale ?")
    assert vus["historique"] == [
        {"role": "user", "content": "je parle du Chili"},
        {"role": "assistant", "content": "Le Chili est un pays."}]


# -- fenetre de conversation ---------------------------------------------------

def test_tours_precedents_excluent_le_tour_en_cours(tmp_path):
    from aura.memoire_conversation import MemoireConversation
    from aura.orchestrateur import RontoGPT
    ia = RontoGPT.__new__(RontoGPT)
    ia._historique = MemoireConversation(chemin=tmp_path / "h.jsonl")
    ia._historique.ajouter("user", "u1")
    ia._historique.ajouter("assistant", "a1")
    ia._historique.ajouter("user", "u2")
    ia._historique.ajouter("assistant", "a2")
    assert ia._tours_precedents() == [{"role": "user", "content": "u1"},
                                      {"role": "assistant", "content": "a1"}]
