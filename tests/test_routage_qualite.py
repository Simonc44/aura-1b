"""Routage qualite — garde-fous issus des defauts trouves par le banc.

Defauts mesures puis corriges (scripts/bench_qualite.py) :
    meta-long  : « Est-ce que tu es une IA ... » partait en dissertation (75 s)
    factuel    : « Quel est le prix du carburant ? » aussi (86 s)
    cache      : un refus en DOUBLE etait servi depuis le cache (0,1 s)
    tetes      : « Je suis desole, mais... » en tete d'une vraie reponse
"""
from aura import llama_cerveau
from aura import orchestrateur as orch


class TestMetaEtReponseCourte:
    def test_meta_long_detectee(self):
        # etait 74,9 s de dissertation au banc
        assert orch._est_meta_ia(
            "Est-ce que tu es une intelligence artificielle vraiment capable "
            "de comprendre ce que tu ecris, ou tu simules seulement ?") is True

    def test_meta_courte_detectee(self):
        assert orch._est_meta_ia("es tu une ia aussi puissn qu'un 8B ou pluas")
        assert orch._est_meta_ia("Tu es plus fort que ChatGPT ?")
        assert orch._est_meta_ia("Tu es conscient de ce que tu dis ?")

    def test_questions_longues_non_meta(self):
        # pas de fausse positif : un essai reste un essai
        assert orch._est_meta_ia(
            "La puissance de la technologie est-elle une menace pour l'homme "
            "moderne ? Redige une dissertation structuree.") is False

    def test_reponse_courte_interdit_la_dissertation(self):
        # « quel ... » / « combien ... » -> reponse courte, jamais le
        # multi-pass (75-86 s mesures au banc)
        for q in ("Quel est le prix du carburant en France aujourd'hui ?",
                  "Alice a 3 ans de plus que Bob. Bob a 2 ans de plus que "
                  "Clara. Clara a 4 ans. Quel age a Alice ?",
                  "Combien de temps dure le voyage ?"):
            assert orch.Aura1B._est_complexe(q) is False, q

    def test_vraie_redaction_toujours_riche(self):
        assert orch.Aura1B._est_complexe(
            "Redige une analyse comparee du solaire et de l'eolien en "
            "France, avec les avantages et les limites de chaque filiere"
        ) is True
        assert orch.Aura1B._est_complexe(
            "Explique pourquoi la photosynthese est indispensable a la vie "
            "sur Terre en developpant son role dans la chaine alimentaire"
        ) is True


class TestRefus:
    REFUS_COURT = "Je suis desole, mais je ne peux pas repondre a cette demande."
    REFUS_PUIS_SUITE = (
        "Je suis desole, mais je ne peux pas repondre a cette demande. "
        + "Pourtant l'analyse de la question montre que la puissance se "
        "mesure en watts et que la notion recouvre des domaines tres "
        "varies : physique d'abord, puis metaphorique ensuite, si bien "
        "que la reponse depend du cadre retenu par l'interlocuteur. ")

    def test_refus_court_identifie(self):
        assert orch._est_refus(self.REFUS_COURT) is True

    def test_reponse_normale_nest_pas_un_refus(self):
        assert orch._est_refus("42") is False
        assert orch._est_refus("Le solaire progresse en Europe.") is False

    def test_refus_puis_suite_restaure(self):
        # la tete parasite est coupee, la vraie reponse reste
        propre = orch._retirer_refus_tete(self.REFUS_PUIS_SUITE)
        assert not propre.lower().startswith("je suis desole")
        assert "analyse de la question" in propre

    def test_refus_sans_suite_est_conserve(self):
        # un refus honnete (hors-ligne, garde-fou) ne doit pas etre vide
        assert orch._retirer_refus_tete(self.REFUS_COURT) == self.REFUS_COURT

    def test_reponse_sans_refus_intouchee(self):
        assert orch._retirer_refus_tete("42") == "42"
        assert orch._retirer_refus_tete("") == ""


class TestCachePropre:
    def test_refus_en_cache_est_bypass(self, monkeypatch):
        """Un refus sali dans le cache ne doit pas etre livre comme une
        reponse : le pipeline complet prend le relais."""
        from aura import llama_cerveau
        question = "Qui est le plus age ? Paul, Marie ou Leo ?"
        monkeypatch.setattr("aura.filtre_instantane.repondre",
                            lambda q: orch._RE_REFUS.pattern and
                            "Je suis desole, mais je ne peux pas repondre.")
        monkeypatch.setattr(orch.Aura1B, "analyser",
                            lambda self, q: {"methodes": ["test"]})
        # ni RAG ni experts ni LLM : on s'arrete au resultat final
        monkeypatch.setattr("aura.rag_binaire.chercher", lambda q: None)
        monkeypatch.setattr(orch.Aura1B, "_expert_special", lambda *a, **k: None)
        monkeypatch.setattr("aura.fast_lang.moteur.repondre", lambda q: None)
        monkeypatch.setattr("aura.calcul_verbal.resoudre", lambda q: None)
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: "Paul est le plus age, 12 ans.")
        r = orch.Aura1B().executer_detaille(question)
        assert r["cerveau_choisi"] != "niveau0-instantane"
        assert "desole" not in r["reponse"].lower()


class TestLibellesDePlan:
    """Le 1B recopie parfois l'en-tete de son prompt dans sa redaction.

    Defaut mesure au banc : riche-puissance KO car « SECTION A REDIGER :
    PARTIE 2 - L'antithese » figurait litteralement dans la reponse (avec,
    en plus, la phrase d'intro recopiee avec un autre premier mot).
    """

    def test_libelle_colle_au_texte_puis_suite(self):
        brut = ("**SECTION A REDIGER : PARTIE 2 - L'antithese : la negation** "
                "Pourtant, la puissance se manifeste d'abord partout.")
        propre = llama_cerveau._sans_libelles(brut)
        assert "SECTION" not in propre
        assert "Pourtant, la puissance" in propre          # vrai texte garde

    def test_libelle_seul_sur_sa_ligne(self):
        brut = ("SECTION A REDIGER : PARTIE 1 : L'AFFIRMATION\n"
                "La these se pose des la premiere phrase.")
        propre = llama_cerveau._sans_libelles(brut)
        assert "SECTION A REDIGER" not in propre
        assert "La these se pose" in propre

    def test_partie_et_connecteurs_retires(self):
        assert "PARTIE 3" not in llama_cerveau._sans_libelles(
            "PARTIE 3 : La synthese\nTout se joue alors.")
        assert "CONNECTEURS" not in llama_cerveau._sans_libelles(
            "CONNECTEURS : neanmoins, donc, pourtant\nIl reste que...")

    def test_texte_sans_libelle_intouche(self):
        t = "La puissance se mesure. Elle se discute aussi."
        assert llama_cerveau._sans_libelles(t) == t

    def test_cache_sali_est_nettoye_au_service(self, monkeypatch):
        """Une vieille entree de cache contenant la balise est nettoyee
        AVANT d'etre servie (pas seulement a l'ecriture)."""
        sale = ("Bonjour. **SECTION A REDIGER : PARTIE 1 : L'AFFIRMATION** "
                "La these se pose des la premiere phrase, sans ambiguite.")
        monkeypatch.setattr("aura.filtre_instantane.repondre", lambda q: sale)
        monkeypatch.setattr(orch.Aura1B, "analyser",
                            lambda self, q: {"methodes": ["test"]})
        monkeypatch.setattr("aura.rag_binaire.chercher", lambda q: None)
        monkeypatch.setattr(orch.Aura1B, "_expert_special", lambda *a, **k: None)
        monkeypatch.setattr("aura.fast_lang.moteur.repondre", lambda q: None)
        monkeypatch.setattr("aura.calcul_verbal.resoudre", lambda q: None)
        r = orch.Aura1B().executer_detaille(
            "La puissance de la technologie est-elle une menace ?")
        assert "SECTION A REDIGER" not in r["reponse"]
        assert "La these se pose" in r["reponse"]


class TestAntiCopieProche:
    """Copie INTER-sections avec un premier mot different : la comparaison
    exacte rate (vu au banc), le fragment de 45+ signes la attrape."""

    P1 = ("Pourtant, cette question souleve un debat complexe qui "
          "concerne notre societe actuelle et ses choix.")
    P2 = ("En definitive, cette question souleve un debat complexe qui "
          "concerne notre societe actuelle et ses choix.")

    def test_phrases_exactement_differentes(self):
        assert self.P1 != self.P2                    # le piege

    def test_sans_copies_retire_la_reprise_debut_different(self):
        propre = llama_cerveau._sans_copies(
            "Transition utile et necessaire vers la suite. " + self.P2,
            self.P1)
        assert "souleve un debat complexe" not in propre
        assert "Transition utile et necessaire" in propre

    def test_sans_doublons_retire_la_copie_proche(self):
        propre = llama_cerveau._sans_doublons(
            self.P1 + " " + self.P2 + " Une chute utile et finale.")
        assert "En definitive" not in propre
        assert "Une chute utile et finale." in propre

    def test_texte_sain_intact(self):
        t = ("Le solaire progresse en Europe. L'eolien reste dependant "
             "du reseau. Une comparaison suppose de comparer les couts "
             "marginaux, pas les subventions declarees.")
        assert llama_cerveau._sans_doublons(t) == t
        assert llama_cerveau._sans_copies(t, "") == t


class TestFinitionDePhrase:
    """Jamais de phrase tronquee en fin de reponse (max_tokens ou coupure
    de disjoncteur au milieu d'un mot : « ... et d' » vu au banc)."""

    def test_queue_morte_coupee(self):
        t = ("Le solaire progresse en Europe. L'eolien reste dependant "
             "du reseau. La puissance de la technologie est")
        coupe = llama_cerveau._finir_phrase(t)
        assert coupe.endswith("reseau.")
        assert "technologie est" not in coupe

    def test_fin_compleete_intacte(self):
        t = "Le solaire progresse en Europe. L'eolien gagne du terrain"
        assert llama_cerveau._finir_phrase(t) == t

    def test_forcer_coupe_au_dernier_point(self):
        t = ("Intro solide et longue. Un debut de phrase coupee par le "
             "stream au moment ou le model")
        assert llama_cerveau._finir_phrase(t, forcer=True) == "Intro solide et longue."

    def test_texte_sans_aucune_ponctuation_intact(self):
        assert llama_cerveau._finir_phrase("un texte sans point", forcer=True) == \
            "un texte sans point"

    def test_queue_arretee_au_milieu_d_un_mot(self):
        # cas reels : max_tokens s'arrete en plein mot (« ... et d' »)
        attendu = "La these se pose sans ambiguite."
        t1 = (attendu + " Prendre conscience de ces risques et d")
        t2 = (attendu + " Prendre conscience de ces risques et d'")
        assert llama_cerveau._finir_phrase(t1) == attendu
        assert llama_cerveau._finir_phrase(t2) == attendu


class TestRepliContexte:
    """Fenetre d'historique trop longue pour la fenetre de contexte :
    erreur vue au banc (Requested tokens exceed context window).

    Deux lignes de defense :
    1. _limiter_messages rogne les VIEUX tours avant l'appel (la
       conversation continue avec les tours recents) ;
    2. si meme systeme + question debordent, le repli repart SANS
       historique au lieu de livrer « [Aura] Erreur ... ».
    """

    def test_limite_les_vieux_tours_avant_l_appel(self, monkeypatch):
        """Prompt trop grand pour n_ctx -> seuls les anciens tours sautent,
        le systeme et la question courante restent (pas de repli brutal)."""
        class FauxLlm:
            _n_ctx = 400                      # fenetre minuscule (test)

            def __init__(self):
                self.appels: list[list] = []

            def tokenize(self, texte, add_bos=False):
                # ~1 token par caractere : le comptage est exact ici
                return list(texte)

            def create_chat_completion(self, messages, **kw):
                self.appels.append(list(messages))
                # se plante si plus de 2 tours d'historique passent
                if len(messages) > 4:
                    raise ValueError("Requested tokens (9999) exceed "
                                     "context window of 400")
                return {"choices": [{"message": {
                    "content": "Reponse bornee."}}]}

        faux = FauxLlm()
        monkeypatch.setattr(llama_cerveau, "_charger", lambda: faux)
        monkeypatch.setattr("aura.serveur_lora._actifs", lambda: False)
        hist = [{"role": "user", "content": "Tour un " + "x" * 150},
                {"role": "assistant", "content": "Reponse un " + "y" * 150},
                {"role": "user", "content": "Tour deux " + "z" * 150},
                {"role": "assistant", "content": "Reponse deux " + "w" * 150}]
        r = llama_cerveau.generer("Et maintenant ?", historique=hist)
        assert r == "Reponse bornee."
        appels = faux.appels[0]
        # systeme + question presents, tours trop vieux rognes
        assert appels[0]["role"] == "system"
        assert appels[-1]["content"].startswith("Et maintenant")
        assert len(appels) < 1 + len(hist) + 1

    def test_debordement_reessaie_sans_historique(self, monkeypatch):
        class FauxLlm:
            def __init__(self):
                self.appels: list[list] = []

            def create_chat_completion(self, messages, **kw):
                self.appels.append(list(messages))
                # systeme + question = 2 messages sans memoire
                if len(messages) > 2:
                    raise ValueError("Requested tokens (9999) exceed "
                                     "context window of 1536")
                return {"choices": [{"message": {
                    "content": "Reponse sans memoire."}}]}

        faux = FauxLlm()
        monkeypatch.setattr(llama_cerveau, "_charger", lambda: faux)
        monkeypatch.setattr("aura.serveur_lora._actifs", lambda: False)
        hist = [{"role": "user", "content": "Question precedente ?"},
                {"role": "assistant", "content": "Reponse precedente."}]
        r = llama_cerveau.generer("Et maintenant ?", historique=hist)
        assert r == "Reponse sans memoire."
        assert len(faux.appels[0]) > 2        # 1er essai : avec historique
        assert len(faux.appels[-1]) == 2      # repli : systeme + question


class TestFluxDirect:
    """Streaming GUI : abonnes aux morceaux + desabonnement garanti."""

    def test_abonnes_recoivent_puis_ne_recoivent_plus(self):
        recus: list[str] = []
        desabonner = llama_cerveau.abonner_flux(recus.append)
        try:
            llama_cerveau.publier_flux("bon")
            llama_cerveau.publier_flux("jour")
        finally:
            desabonner()
        assert recus == ["bon", "jour"]
        llama_cerveau.publier_flux("apres")      # desabonne : ignore
        assert recus == ["bon", "jour"]

    def test_abonne_en_erreur_ne_tue_pas_la_publication(self):
        recus: list[str] = []

        def _foireux(_m: str) -> None:
            raise RuntimeError("abonne ko")

        d1 = llama_cerveau.abonner_flux(_foireux)
        d2 = llama_cerveau.abonner_flux(recus.append)
        try:
            llama_cerveau.publier_flux("ok")
        finally:
            d1()
            d2()
        assert recus == ["ok"]


class TestMemoirePersistante:
    """Etat civil entre sessions + outil local « lis le fichier »."""

    @staticmethod
    def _memoire(tmp_path):
        from aura.memoire_conversation import MemoireConversation
        return MemoireConversation(chemin=tmp_path / "conv.jsonl")

    def test_etat_civil_survit_une_restauration(self, tmp_path):
        m = self._memoire(tmp_path)
        m.ajouter("user", "je m'appelle Simon")
        assert m.etat().get("utilisateur") == "Simon"
        # nouvelle session (nouvelle instance, meme disque) -> meme etat
        assert self._memoire(tmp_path).etat().get("utilisateur") == "Simon"

    def test_vider_efface_aussi_etat_disque(self, tmp_path):
        m = self._memoire(tmp_path)
        m.ajouter("user", "je m'appelle Simon")
        m.vider()
        assert m.etat() == {}
        assert self._memoire(tmp_path).etat() == {}

    def test_outil_fichier_inactif_sur_question_normale(self, tmp_path):
        m = self._memoire(tmp_path)
        assert m.lire_fichier("quel temps fait-il ?") is None

    def test_outil_fichier_lit_le_texte(self, tmp_path):
        cible = tmp_path / "notes.txt"
        cible.write_text("code du projet : 42", encoding="utf-8")
        m = self._memoire(tmp_path)
        contenu = m.lire_fichier(f"lis le fichier {cible}")
        assert contenu is not None and "42" in contenu

    def test_outil_fichier_signale_lintrouvable(self, tmp_path):
        m = self._memoire(tmp_path)
        r = m.lire_fichier(f"lis le fichier {tmp_path / 'absent.txt'}")
        assert r is not None and "introuvable" in r

    def test_outil_fichier_refuse_le_binaire(self, tmp_path):
        binaire = tmp_path / "data.bin"
        binaire.write_bytes(b"\x00\x01\x02taille quelconque")
        m = self._memoire(tmp_path)
        r = m.lire_fichier(f"lis le fichier {binaire}")
        assert r is not None and "binaire" in r
