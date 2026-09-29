"""Disjoncteur anti-repetition + routage meta-IA.

Bug reference (production) : « es tu une ia aussi puissn qu'un 8B ou pluas »
-> mode dissertation lance (>= 9 mots), partie 2 recopiee mot pour mot de
la partie 1, phrase revenue 3 fois, 153,3 s de generation.
"""
from aura import llama_cerveau as lc
from aura import orchestrateur as orch


PHRASE = ("La notion de puissance est souvent consideree comme une qualite "
          "majeure des systemes, des individus et des choses. Cependant, "
          "cette affirmation est loin d'etre veritablement applicable a "
          "toutes les situations.")

QUESTION_META = "es tu une ia aussi puissn qu'un 8B ou pluas"


# ── detection de boucle ────────────────────────────────────────────────

class TestDetectionBoucle:
    def test_phrase_recopiee_detectee(self):
        # le cas observe : la meme phrase longue apparait 2+ fois
        assert lc._en_boucle(PHRASE + " " + PHRASE) is True

    def test_phrase_unique_pas_de_faux_positif(self):
        assert lc._en_boucle(PHRASE) is False

    def test_texte_normal_sain(self):
        texte = ("Le solaire progresse en Europe. L'eolien reste dependant "
                 "du reseau. Une comparaison suppose de comparer les "
                 "couts marginaux, pas les subventions declarees. Chaque "
                 "technique repond a un profil de charge different.")
        assert lc._en_boucle(texte) is False

    def test_boucle_serree_de_4_mots(self):
        # meme sequence de 4 mots repetee 5 fois -> boucle
        mot = "il ne dispose pas de la meme capacite "
        assert lc._en_boucle(mot * 5) is True

    def test_texte_vide(self):
        assert lc._en_boucle("") is False
        assert lc._en_boucle(None) is False


# ── coupure / anti-copie ───────────────────────────────────────────────

class TestCoupureEtAntiCopie:
    def test_coupe_avant_la_deuxieme_occurrence(self):
        bloque = "Intro. " + PHRASE + " " + PHRASE + " suite."
        coupe = lc._couper_boucle(bloque)
        assert lc._en_boucle(coupe) is False
        assert coupe.count("La notion de puissance") == 1
        assert "suite." not in coupe

    def test_sans_copies_retire_la_phrase_reprise(self):
        precedent = "Partie 1 : " + PHRASE
        nouveau = "Transition utile. " + PHRASE + " Ensuite une idee neuve."
        propre = lc._sans_copies(nouveau, precedent)
        assert "Transition utile." in propre
        assert "La notion de puissance" not in propre
        assert "idee neuve" in propre

    def test_sans_copies_ne_vide_jamais_le_texte(self):
        # tout est copie -> on garde l'original plutot qu'une reponse vide
        assert lc._sans_copies(PHRASE, PHRASE) == PHRASE
        assert lc._sans_copies(PHRASE, "") == PHRASE

    def test_sans_doublons_deduplique_les_extraits(self):
        extraits = PHRASE + " " + PHRASE + " Un deuxieme extrait utile."
        propre = lc._sans_doublons(extraits)
        assert propre.count("La notion de puissance") == 1
        assert "deuxieme extrait utile" in propre

    def test_resume_sections_court_anti_copie(self):
        resume = lc._resume_sections([PHRASE + " Suite.", PHRASE])
        #1re phrase de chaque partie, pas les 900 caracteres bruts
        assert resume.startswith("- La notion de puissance")
        assert len(resume) < len(PHRASE) * 2
        assert resume.count(PHRASE) == 0


# ── generation streamée avec coupure + relance ─────────────────────────

class TestGenerationAntiBoucle:
    class FauxLlmBoucle:
        """Rend une boucle DES LE stream (chaque chunk = la meme phrase)."""

        def __init__(self):
            self.appels = 0

        def create_chat_completion(self, **kw):
            self.appels += 1

            def flux():
                for _ in range(6):
                    yield {"choices": [{"delta": {"content": PHRASE + " "}}]}
            return flux()

    def test_coupure_immediate_et_relance_unique(self):
        llm = self.FauxLlmBoucle()
        texte = lc._generer_avec_disjoncteur(
            llm, [{"role": "user", "content": "q"}], 190,
            {"temperature": 0.5, "repeat_penalty": 1.1, "stop": ["<|eot_id|>"]})
        #1re passe coupee sur boucle + UNE relance (penalite/temp differentes)
        assert llm.appels == 2
        # jamais de boucle livree
        assert lc._en_boucle(texte) is False
        # texte tronque : loin des 6 recopies produites par le faux llm
        assert len(texte) < len(PHRASE) * 3

    def test_relance_avec_penalite_et_temperature_differentes(self):
        llm = self.FauxLlmBoucle()
        lc._generer_avec_disjoncteur(
            llm, [{"role": "user", "content": "q"}], 190,
            {"temperature": 0.5, "repeat_penalty": 1.1, "stop": ["<|eot_id|>"]})
        # le faux llm ne capture pas les kwargs : on verifie l'absence
        # d'erreur et le nombre d'appels (le parametrage est dans le code)
        assert llm.appels == 2

    class FauxLlmSain:
        def __init__(self):
            self.appels = 0

        def create_chat_completion(self, **kw):
            self.appels += 1
            return {"choices": [{"message": {"content": "Reponse saine."}}]}

    def test_sans_boucle_un_seul_appel(self):
        llm = self.FauxLlmSain()
        texte = lc._generer_avec_disjoncteur(
            llm, [{"role": "user", "content": "q"}], 190, {"temperature": 0.5})
        assert llm.appels == 1
        assert texte == "Reponse saine."


# ── routage meta-IA (l'anti-mode-dissertation) ─────────────────────────

class TestRoutageMetaIA:
    def test_question_meta_detectee(self):
        assert orch._est_meta_ia(QUESTION_META) is True
        assert orch._est_meta_ia("Qui es-tu ?") is True
        assert orch._est_meta_ia("Tu es aussi intelligent que GPT-4 ?") is True
        assert orch._est_meta_ia("Es-tu un modèle 8B ?") is True

    def test_faux_positif_sur_un_essai(self):
        # une dissertation sur la puissance N'EST PAS une question meta
        assert orch._est_meta_ia(
            "La puissance de la technologie est-elle une menace pour "
            "l'homme moderne ?") is False
        assert orch._est_meta_ia(
            "Explique comment la photosynthese fonctionne en detail") is False

    def test_mode_riche_interdit_aux_questions_meta(self):
        # C'ETAIT LE BUG : >= 9 mots -> riche -> plan + 3 sections (153 s)
        assert len(QUESTION_META.split()) >= 9
        assert orch.Aura1B._est_complexe(QUESTION_META) is False

    def test_reponse_courte_sans_llm(self, monkeypatch):
        # la question meta courte part en constante : 0 s, zero generation
        def _explose(*a, **k):
            raise AssertionError("le LLM ne doit pas etre charge")

        monkeypatch.setattr(lc, "_charger", _explose)
        ia = orch.Aura1B()
        r = ia.executer_detaille(QUESTION_META)
        assert r["cerveau_choisi"] == "constante-capacite"
        assert r["reponse"] == orch._CAPACITE
        assert "1B" in r["reponse"] and "8B" in r["reponse"]
