"""Pont lm-eval : commande correcte, refus des taches non mesurables.

Le pont ne doit JAMAIS importer lm_eval a l'import du module (sinon toute la
suite de tests dependrait d'une dependance lourde) : il se contente de
construire l'argv de `python -m lm_eval`.

Les deux tests de `type_sortie` se lisaient dans une installation lm-eval et
se SAUTAIENT quand elle etait absente — c'est-a-dire toujours, `lm-eval` (et
son torch) n'etant pas une dependance du projet. Ils tournent desormais sur un
dossier de taches qui reproduit les formes reelles (groupe sans fichier a son
nom, `include:` qui porte l'output_type, homonyme de FICHIER qui note
autrement) : zero saut, et la regle verrouillee est explicite au lieu de
dependre de la version de lm-eval du jour.
"""
import sys
import types
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))

import lm_eval_aura as le  # noqa: E402


def test_base_url_brut_et_chat():
    assert le.base_url("127.0.0.1:8757") == \
        "http://127.0.0.1:8757/v1/completions"
    assert le.base_url("127.0.0.1:8757", chat=True) == \
        "http://127.0.0.1:8757/v1/chat/completions"


def test_commande_par_defaut_chat_sans_recharger_de_poids():
    cmd = le.construire_commande("gsm8k", limite=50, sortie="r.json")
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    assert "--apply_chat_template" in cmd
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "base_url=http://" in model_args
    assert "/v1/chat/completions" in model_args
    assert "tokenizer_backend=None" in model_args
    assert cmd[cmd.index("--tasks") + 1] == "gsm8k"
    assert cmd[cmd.index("--limit") + 1] == "50"
    assert cmd[cmd.index("--output_path") + 1] == "r.json"


def test_mode_brut_sans_gabarit_de_conversation():
    cmd = le.construire_commande("gsm8k", brut=True)
    assert cmd[cmd.index("--model") + 1] == "local-completions"
    assert "--apply_chat_template" not in cmd
    assert "/v1/completions" in cmd[cmd.index("--model_args") + 1]


def test_fewshot_transmis_seulement_si_demande():
    assert "--num_fewshot" not in le.construire_commande("gsm8k")
    cmd = le.construire_commande("gsm8k", fewshot=8)
    assert cmd[cmd.index("--num_fewshot") + 1] == "8"


def test_sans_limite_ni_sortie_les_options_sont_absentes():
    cmd = le.construire_commande("gsm8k")
    assert "--limit" not in cmd and "--output_path" not in cmd


def test_adresse_suit_la_variable_du_projet(monkeypatch):
    monkeypatch.setenv("AURA_SERVEUR_ADRESSE", "127.0.0.1:9999")
    assert le.base_url() == "http://127.0.0.1:9999/v1/completions"
    assert le.base_url(chat=True) == \
        "http://127.0.0.1:9999/v1/chat/completions"


def _dossier_taches(monkeypatch, tmp_path: Path) -> Path:
    """Un dossier de taches qui reproduit les FORMES de lm-eval.

    Ce n'est pas une imitation decorative : chaque fichier existe pour une
    regle de `scripts/lm_eval_aura.py` qui a deja ete prise en defaut.

    * `mmlu` n'a **aucun** fichier a son nom — seulement `_mmlu.yaml`, qui
      declare `group: mmlu`. Chercher `mmlu.yaml` le rendait invisible, donc
      mesurable : c'est le bug d'origine.
    * son `output_type` n'est PAS sur le groupe : il faut suivre `task:`,
      puis `include:` jusqu'au gabarit qui le porte.
    * `_mmlu_stem.yaml` existe DEUX fois. La copie de `01_variante/` trie
      avant celle de `default/` : si la resolution retombait sur le nom de
      FICHIER, elle gagnerait — et elle note en generation, donc un groupe a
      choix multiple passerait pour mesurable. Seul le nom DECLARE decides.
    """
    racine = tmp_path / "taches"
    (racine / "default").mkdir(parents=True)
    (racine / "01_variante").mkdir(parents=True)
    (racine / "gsm8k.yaml").write_text(
        "task: gsm8k\noutput_type: generate_until\n", encoding="utf-8")
    (racine / "arc_easy.yaml").write_text(
        "task: arc_easy\noutput_type: loglikelihood\n", encoding="utf-8")
    (racine / "default" / "_mmlu.yaml").write_text(
        "group: mmlu\ntask:\n  - mmlu_abstract_algebra\n", encoding="utf-8")
    (racine / "default" / "mmlu_abstract_algebra.yaml").write_text(
        "task: mmlu_abstract_algebra\ninclude: _default_template_yaml\n",
        encoding="utf-8")
    (racine / "default" / "_default_template_yaml.yaml").write_text(
        "output_type: multiple_choice\n", encoding="utf-8")
    (racine / "default" / "_mmlu_stem.yaml").write_text(
        "group: mmlu_stem\ntask:\n  - mmlu_abstract_algebra\n", encoding="utf-8")
    (racine / "01_variante" / "_mmlu_stem.yaml").write_text(
        "task: une_autre_tache\noutput_type: generate_until\n", encoding="utf-8")

    # INJECTION par le contrat de cache du module : `type_sortie` lit `_YAMLS`,
    # bati une seule fois depuis `dossier_taches()`. On pose la liste du
    # dossier de test et on laisse l'index des noms se reconstruire.
    monkeypatch.setattr(le, "_YAMLS", le._yamls(racine))
    monkeypatch.setattr(le, "_INDEX_NOMS", None)
    return racine


def test_type_sortie_lit_les_yaml_de_taches(monkeypatch, tmp_path):
    _dossier_taches(monkeypatch, tmp_path)
    assert le.type_sortie("gsm8k") == "generate_until"
    # arc_easy / hellaswag demandent les logprobs du prompt : non mesurables.
    assert le.taches_loglikelihood("gsm8k,arc_easy") == ["arc_easy"]


def test_refus_des_groupes_et_pas_seulement_des_taches(monkeypatch, tmp_path):
    """`mmlu` est un GROUPE (`_mmlu.yaml` porte `group: mmlu`), pas un fichier
    de tache : chercher `mmlu.yaml` le rendait invisible, donc il passait le
    refus et aurait produit un score de hasard.

    Et le nom DECLARE doit primer sur le nom de FICHIER : trois fichiers
    s'appellent `_mmlu_stem.yaml` (variantes llama3, llama3_cot, et le vrai
    mmlu), et le tri par chemin tombait sur une variante qui note en
    generation — un groupe a choix multiple passait alors pour mesurable.
    """
    _dossier_taches(monkeypatch, tmp_path)
    assert le.type_sortie("mmlu") == "multiple_choice"
    assert le.type_sortie("mmlu_stem") == "multiple_choice"
    assert "mmlu" in le.taches_loglikelihood("gsm8k,mmlu")
    # et le generateur reste mesurable : on n'a pas refuse tout le monde
    assert le.type_sortie("gsm8k") == "generate_until"
    assert le.taches_loglikelihood("gsm8k") == []


def test_message_actionnable_si_lm_eval_absent(monkeypatch, capsys):
    monkeypatch.setattr(le, "lm_eval_installe", lambda: False)
    assert le.main([]) == 2
    assert "pip install lm-eval" in capsys.readouterr().err


def test_refus_des_taches_a_loglikelihood(monkeypatch, capsys):
    monkeypatch.setattr(le, "lm_eval_installe", lambda: True)
    monkeypatch.setattr(le, "taches_loglikelihood", lambda _t: ["arc_easy"])
    assert le.main(["--taches", "arc_easy"]) == 2
    err = capsys.readouterr().err
    assert "refus" in err and "27174" in err


def test_loglikelihood_force_avertit_mais_poursuit(monkeypatch, capsys):
    monkeypatch.setattr(le, "lm_eval_installe", lambda: True)
    monkeypatch.setattr(le, "taches_loglikelihood", lambda _t: ["arc_easy"])
    monkeypatch.setattr(le, "subprocess", types.SimpleNamespace(
        run=lambda *a, **k: types.SimpleNamespace(returncode=0)))
    assert le.main(["--taches", "arc_easy", "--loglikelihood"]) == 0
    assert "NON valides" in capsys.readouterr().err
