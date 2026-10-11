"""Tests des experts deterministes calcul verbal et tri transitif.

Ces experts interceptent les petits problemes racontes que ni le
niveau 0 (pas d'expression mathematique) ni le mini-SAT (formalisme
trop lourd) ne savaient resoudre — le 1B repondait alors hors-sujet.
"""
from aura import calcul_verbal, tri_transitif
from aura.orchestrateur import RontoGPT


class TestCalculVerbal:
    def test_pommes(self):
        assert calcul_verbal.resoudre(
            "si j ai 3 pommes et que j en mange 1 combien il m en reste") == "2"

    def test_argent_perdu(self):
        assert calcul_verbal.resoudre(
            "j avais 10 euros j en perds 4 combien me reste t il") == "6"

    def test_achats_enumeration(self):
        assert calcul_verbal.resoudre(
            "j achete 5 cahiers et 3 stylos combien en tout") == "8"

    def test_gain(self):
        assert calcul_verbal.resoudre(
            "j avais 12 billes j en gagne 7 combien en tout") == "19"

    def test_pas_de_question_pas_de_calcul(self):
        assert calcul_verbal.resoudre("qu'est ce que la photosynthese") is None
        assert calcul_verbal.resoudre("raconte moi une blague") is None

    def test_abstention_si_negatif(self):
        # 3 - 5 = -2 : quantite physique absurde -> abstention (None),
        # le cerveau repondra — le traducteur n'invente pas.
        assert calcul_verbal.resoudre(
            "j ai 3 pommes j en mange 5 combien il m en reste") is None

    def test_depart_par_contient(self):
        # « contient 24 » n'etait pas reconnu comme valeur de depart
        assert calcul_verbal.resoudre(
            "Un paquet contient 24 biscuits, j'en mange 9, combien en "
            "reste-t-il ?") == "15"

    def test_verbe_coupe(self):
        # « coupe » manquait au lexique des retraits
        assert calcul_verbal.resoudre(
            "Un gateau a 12 parts, on en coupe 5, combien reste-t-il ?") \
            == "7"

    def test_moitie_puis_retrait(self):
        # etait 10 : le « 2 » de « puis 2 a Clara » etait pris pour
        # l'operation et la moitie etait ignoree -> 12/2-2 = 4
        assert calcul_verbal.resoudre(
            "Alice a 12 bonbons, elle en donne la moitie a Bob puis 2 a "
            "Clara. Combien lui en reste-t-il ?") == "4"

    def test_quantite_avant_le_verbe(self):
        # « 17 descendent et 9 montent » : en francais la quantite est
        # AVANT le verbe (etait rendu 36 = 45-9 au lieu de 37)
        assert calcul_verbal.resoudre(
            "Un bus contient 45 passagers. 17 descendent et 9 montent. "
            "Combien de passagers reste-t-il ?") == "37"

    def test_branche_dans_le_pipeline(self, monkeypatch):
        """La question pommes ne doit JAMAIS atteindre le cerveau."""
        from aura import llama_cerveau
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("calcul verbal doit repondre")))
        ia = RontoGPT()
        r = ia.executer_detaille(
            "si j ai 3 pommes et que j en mange 1 combien il m en reste")
        assert r["cerveau_choisi"] == "calcul-verbal"
        assert r["reponse"] == "2"


class TestCalculVerbalEtendu:
    """Familles ajoutees le 7 oct. 2026, toutes trouvees par le SOUS-BANC.

    Elles etaient listees comme « tombe sur le LLM » (~40 s, valeur non
    prouvee) alors que ce sont des calculs a une etape que le lexique ne
    couvrait pas.
    """

    def test_ecart_entre_deux_quantites(self):
        assert calcul_verbal.resoudre(
            "Marie a 14 ans et son frere a 9 ans, quel est l'ecart d'age ?") \
            == "5"

    def test_groupe_rangee_de(self):
        assert calcul_verbal.resoudre(
            "Il y a 6 rangees de 7 chaises, combien de chaises ?") == "42"

    def test_groupe_fois(self):
        assert calcul_verbal.resoudre(
            "Paul a 3 fois 8 billes, combien en a-t-il ?") == "24"

    def test_groupe_puis_retrait(self):
        """La reduction du groupe laisse la chaine continuer : 5x12 - 7."""
        assert calcul_verbal.resoudre(
            "Il y a 5 boites de 12 oeufs. J'en casse 7. Combien d'oeufs "
            "entiers reste-t-il ?") == "53"

    def test_chacun(self):
        assert calcul_verbal.resoudre(
            "Il y a 9 eleves et chacun a 4 cahiers, combien de cahiers ?") \
            == "36"

    def test_meme_unite_enchaisnee(self):
        assert calcul_verbal.resoudre(
            "Un cycliste roule 18 km puis 9 km, combien en tout ?") == "27"

    def test_distribue_avec_pronom_en(self):
        # « j'en distribue 23 » = on en RETIRE 23
        assert calcul_verbal.resoudre(
            "J'ai 50 cartes, j'en distribue 23, combien m'en reste-t-il ?") \
            == "27"

    def test_distribue_sans_pronom_nest_pas_un_retrait(self):
        """« distribue 12 bonbons entre 4 amis » est une DIVISION.

        Le pronom « en » est exige pour lire un retrait : sans cette
        garde, la regle rendrait 8 au lieu de 3.
        """
        assert calcul_verbal.resoudre(
            "J'ai 12 bonbons et je les distribue entre 4 amis, combien "
            "chacun ?") != "8"

    def test_prix_unitaire_nest_pas_un_chacun(self):
        """« 3 livres a 18 euros chacun » ne doit pas etre lu 3 x 18.

        C'est le terrain de calcul_achat (budget - cout), place AVANT
        calcul_verbal dans le registre.
        """
        assert calcul_verbal.resoudre(
            "J'ai 100 euros. J'achete 3 livres a 18 euros chacun.") is None

    def test_pizza_partagee_abstention(self):
        """« partagee entre 4 » est une division : calcul_taux, pas le verbal."""
        assert calcul_verbal.resoudre(
            "Une pizza de 8 parts est partagee entre 4 personnes. Combien de "
            "parts chacune ?") is None


class TestTriTransitif:
    def test_chaine_age(self):
        assert tri_transitif.repondre(
            "resous cette enigme logique : Paul est plus age que Marie, "
            "Marie est plus age que Leo, qui est le plus age ?") == "Paul"

    def test_signes(self):
        assert tri_transitif.repondre(
            "A > B, B > C, qui est le plus grand ?") == "A"

    def test_inversion_petit(self):
        assert tri_transitif.repondre(
            "Leo est plus jeune que Marie et Marie est plus jeune que Paul. "
            "Qui est le plus jeune ?") == "Leo"

    def test_une_seule_relation_insuffisante(self):
        assert tri_transitif.repondre(
            "Paul est plus age que Marie") is None

    def test_sans_question_pas_dexperte(self):
        assert tri_transitif.repondre(
            "qu est ce que la photosynthese") is None

    def test_branche_dans_le_pipeline(self, monkeypatch):
        """L'enigme doit etre resolue par le tri, pas par le 1B.

        Le cache niveau 0 est neutralise : sinon une execution precedente
        (test unitaire, question reelle) repond avant le pipeline —
        comportement voulu en prod, mais ce test verifie le ROUTAGE.
        """
        from aura import llama_cerveau, filtre_instantane
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)
        monkeypatch.setattr(llama_cerveau, "generer",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("tri transitif doit repondre")))
        ia = RontoGPT()
        r = ia.executer_detaille(
            "Paul est plus age que Marie, Marie est plus age que Leo, "
            "qui est le plus age ?")
        assert r["cerveau_choisi"] == "tri-transitif"
        assert r["reponse"] == "Paul"

    def test_cycle_abstention(self):
        # A > B et B > A : contradiction -> None (jamais de reponse au
        # hasard face a une enigme incoherente)
        assert tri_transitif.trier([("A", "B"), ("B", "A")]) is None
