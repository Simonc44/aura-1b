"""Tests du few-shot agressif de redaction abstraite (these/antithese/synthese).

Le 1B ne doit pas inventer la structure d'une dissertation : il recoit
un exemple complet a imiter, un plan contraint (GBNF) et un ROLE
rhetorique par section (affirmer / contrer / depasser).
"""
from aura import llama_cerveau


class TestPromptDissertation:
    def test_exemple_complet_present(self):
        ex = llama_cerveau._EXEMPLE_DISSERTATION
        assert "these" in ex and "antithese" in ex and "synthese" in ex
        assert "Q:" in ex and "R:" in ex   # un vrai exemple, pas une consigne vide

    def test_prompt_plan_exige_tas(self):
        p = llama_cerveau._PROMPT_PLAN.format(contexte="ctx", question="q")
        assert "THE" in p.upper() or "these" in p.lower()
        assert "ANTITH" in p.upper()
        assert "SYNTHE" in p.upper()
        # l'exemple vit dans le message SYSTEME (anti-contamination de theme)
        sys_p = llama_cerveau._SYSTEME_PLAN
        assert "these" in sys_p and "antithese" in sys_p
        assert "REGLE ABSOLUE" in sys_p          # regle anti-copie
        assert "Ne reprends JAMAIS" in sys_p

    def test_roles_section_complets(self):
        r = llama_cerveau._ROLES_SECTION
        assert set(r) == {1, 2, 3}
        assert "AFFIRMER" in r[1]
        assert "CONTRER" in r[2]
        assert "DEPASSER" in r[3]

    def test_prompt_section_inclut_le_role(self):
        p = llama_cerveau._PROMPT_SECTION.format(
            numero=2, total=3, titre="L'antithese", question="q ?",
            role_section=llama_cerveau._ROLES_SECTION[2],
            contexte_court="faits", memoire="deja ecrit")
        assert "CONTRER" in p
        assert "Pourtant" in p or "Cependant" in p

    def test_prompt_style_interdit_la_conclusion_molle(self):
        p = llama_cerveau._PROMPT_STYLE.format(
            plan="PARTIE 1 : x", question="q", contexte_court="c")
        assert "pour et du contre" in p    # explicitement interdit

    def test_role_hors_champ_retombe_sur_synthese(self):
        # une 4e section eventuelle herite du role synthese (min(num, 3))
        p = llama_cerveau._PROMPT_SECTION.format(
            numero=4, total=4, titre="Fin", question="q",
            role_section=llama_cerveau._ROLES_SECTION.get(
                min(4, 3), llama_cerveau._ROLES_SECTION[3]),
            contexte_court="c", memoire="m")
        assert "DEPASSER" in p
