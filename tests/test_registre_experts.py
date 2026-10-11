"""Le registre d'experts est la source de verite de l'ORDRE — et il est verifie.

Trois verrous, chacun contre un mode de defaillance observe :

1. INVARIANT DE GENRE — un etage approximatif ne peut pas etre declare avant
   un etage plus fiable. C'est le defaut qui a fait passer le cache semantique
   avant les solveurs exacts (corrige le 6 oct. 2026).
2. BRANCHEMENT EXHAUSTIF — tout expert declare au registre doit avoir un appel
   dans `orchestrateur._appel_expert`. Sans ce test, un expert du registre
   disparaitrait en silence (il ne serait jamais consulte).
3. ORDRE REEL — la cascade doit consulter les experts DANS L'ORDRE DECLARE.
   C'est le verrou qui compte : il compare le registre a l'execution, au lieu
   de comparer le registre a lui-meme.
4. POSITION DES PORTES — une PORTE ne propose pas de reponse mais peut
   ARRETER la cascade (la clarification interactive). Sa position est
   declaree (`PORTES` -> « apres-experts ») et verifiee par le comportement :
   une couche qui sait repondre doit gagner, meme quand la question est
   exactement celle qu'une porte voudrait clarifier.
"""
import pytest

from aura import registre_experts as R
from aura.orchestrateur import RontoGPT

# Question neutre : ni riche, ni puzzle, ni identite, ni meta-IA — pour que
# TOUS les experts (gardes comprises) soient consultes.
_Q = "Decris le vent du nord."


@pytest.fixture
def _llm_muet(monkeypatch):
    from aura import llama_cerveau
    monkeypatch.setattr(llama_cerveau, "generer", lambda *a, **k: "[GENERE]")


class TestPortes:
    def test_portes_saines(self):
        assert R.verifier_portes() == []
        assert "clarification" in [p.nom for p in R.PORTES]

    def test_position_declaree(self):
        assert R.porte("clarification").position == R.APRES_EXPERTS
        assert R.APRES_EXPERTS in R.POSITIONS

    def test_position_inconnue_est_signalee(self, monkeypatch):
        monkeypatch.setattr(R, "PORTES",
                            (R.Porte("porte-fantome", "nulle-part", "x"),))
        assert R.verifier(), "une position inconnue doit etre signalee"

    def test_une_porte_ne_double_pas_un_expert(self, monkeypatch):
        monkeypatch.setattr(R, "PORTES",
                            (R.Porte("fast-lang", R.APRES_EXPERTS, "x"),))
        assert R.verifier(), "un nom a la fois expert et porte doit etre signale"

    def test_la_porte_est_apres_les_experts(self, monkeypatch):
        """ORDRE REELLEMENT EXECUTE : une couche qui repond gagne.

        Si la porte etait posee trop tot, une question que `fast_lang` sait
        traiter recevrait « precise ta demande » — le defaut du cache
        semantique (6 oct. 2026) a l'envers.
        """
        from aura import clarification, fast_lang
        from aura.orchestrateur import RontoGPT

        monkeypatch.setattr(fast_lang.moteur, "repondre",
                            lambda q: "resolu")
        besoin = clarification.Besoin(question="q", famille="test",
                                      manque=("objet",),
                                      questions=("Sur quoi ?",), raison="t")
        monkeypatch.setattr(clarification, "detecter",
                            lambda q, a_contexte=False: besoin)
        ia = RontoGPT()
        assert ia.executer_detaille("Explique")["cerveau_choisi"] == "fast-lang"


class TestCoherenceDuSousBanc:
    """La mesure de provenance doit connaitre TOUT expert qui ne genere pas.

    `scripts/sous_banc.py` tenait la liste des tiers « exacts » RECOPIEE a
    la main : le jour ou `problemes-arithmetiques` est ne, elle ne l'a pas
    connu et le banc a compte 3 victoires comme de simples changements de
    couche. La liste est desormais derivee du registre — ce test verrouille
    la derivation (et interdit d'y reintroduire a la main un tier qui
    appelle le modele).
    """

    @staticmethod
    def _sous_banc():
        import sys
        from pathlib import Path
        scripts = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import sous_banc
        return sous_banc

    def test_tout_exact_du_registre_est_dans_la_mesure(self):
        sb = self._sous_banc()
        tiers = sb._tiers_exacts()
        for e in R.EXPERTS:
            if e.genre in (R.EXACT, R.VERIFIE) and e.nom != "experts-speciaux":
                assert e.nom in tiers, (
                    "%s resout sans generer mais le sous-banc ne le sait "
                    "pas : ses victoires seraient comptees comme de "
                    "simples changements de couche" % e.nom)

    def test_aucun_tier_generatif_dans_le_couvert(self):
        sb = self._sous_banc()
        assert not (sb._tiers_exacts() & sb._TIERS_GENERATIFS)

    def test_la_derivation_est_paresseuse(self):
        """`_tiers_exacts` ne doit PAS etre appele a l'import du module.

        Le calcul importe `aura`, qui fige `filtre_instantane._FICHIER`
        depuis `AURA_FILTRE_CACHE` — variable que `main()` ne pose qu'apres.
        Appele au niveau module, le banc lisait le cache REEL du projet et
        sortait une reference de 97/100 entierement fausse.
        """
        sb = self._sous_banc()
        assert not isinstance(getattr(sb, "_TIERS_EXACTS", None), set), (
            "_TIERS_EXACTS ne doit plus exister au niveau module : la "
            "derivation est paresseuse (cf. sous_banc._tiers_exacts)")


class TestInvariantDeGenre:
    def test_registre_sain(self):
        assert R.verifier() == []

    def test_aucun_flou_avant_un_exact(self):
        rangs = [R.expert(n).rang for n in R.noms()]
        assert rangs == sorted(rangs), (
            "l'ordre declare n'est pas croissant en fiabilite : %s"
            % [(n, R.expert(n).genre) for n in R.noms()])

    def test_verifier_refuse_un_ordre_inverse(self, monkeypatch):
        """Le controleur n'est pas complaisant : ordre inverse -> refuse."""
        inverse = tuple(reversed(R.EXPERTS))
        monkeypatch.setattr(R, "EXPERTS", inverse)
        pbs = R.verifier()
        assert pbs, "un registre inverse (flou avant exact) doit etre signale"
        assert any("avant" in p for p in pbs)

    def test_les_gardes_declarees_existent(self):
        for nom in R.GARDES:
            assert nom in R.noms()


class TestBranchement:
    def test_chaque_expert_est_branche(self, monkeypatch, _llm_muet):
        """`_appel_expert` doit connaitre TOUS les experts du registre.

        Un expert declare mais non branche est un bug silencieux : il
        n'apparaitrait jamais dans la cascade. Le branchement leve KeyError
        pour la rendre bruyante — ce test la provoque exprès.
        """
        ia = RontoGPT()
        for nom in R.noms():
            try:
                ia._appel_expert(nom, _Q, None, None, 0.0)
            except KeyError as e:
                pytest.fail(f"expert declare au registre mais non branche : {e}")

    def test_expert_inconnu_leve(self):
        ia = RontoGPT()
        with pytest.raises(KeyError):
            ia._appel_expert("expert-inexistant", _Q, None, None, 0.0)


class TestOrdreReel:
    def test_la_cascade_suit_le_registre(self, monkeypatch, _llm_muet):
        """L'ORDRE EXECUTE doit etre l'ordre DECLARE (les gardes a part)."""
        from aura import (calcul_achat, calcul_verbal, calcul_taux,
                          chaines_numeriques, fast_lang, filtre_instantane,
                          micro_faits, problemes_arithmetiques, rag_binaire)

        vus: list[str] = []

        def _trace(nom):
            def _f(*a, **k):
                vus.append(nom)
                return None
            return _f

        monkeypatch.setattr(fast_lang.moteur, "repondre", _trace("fast-lang"))
        monkeypatch.setattr(calcul_achat, "resoudre", _trace("calcul-achat"))
        monkeypatch.setattr(calcul_verbal, "resoudre", _trace("calcul-verbal"))
        monkeypatch.setattr(calcul_taux, "resoudre", _trace("calcul-taux"))
        monkeypatch.setattr(chaines_numeriques, "resoudre",
                            _trace("chaines-numeriques"))
        monkeypatch.setattr(problemes_arithmetiques, "resoudre",
                            _trace("problemes-arithmetiques"))
        monkeypatch.setattr(micro_faits, "resoudre", _trace("micro-faits"))
        monkeypatch.setattr(RontoGPT, "_expert_special", _trace("experts-speciaux"))
        monkeypatch.setattr(rag_binaire, "chercher", _trace("rag-binaire"))
        monkeypatch.setattr(filtre_instantane, "repondre_cache",
                            _trace("cache-semantique"))
        # le niveau 0 EXACT doit lui aussi etre muet (sinon rien n'est consulte)
        monkeypatch.setattr(filtre_instantane, "repondre", lambda q: None)

        RontoGPT().executer_detaille(_Q)
        assert tuple(vus) == R.noms(), (
            "la cascade ne suit pas le registre.\n  declare : %s\n  execute : %s"
            % (R.noms(), tuple(vus)))
