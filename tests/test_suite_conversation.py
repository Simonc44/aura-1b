"""Une demande de reprise (« répète ») depend de la conversation : elle ne doit
jamais etre servie depuis un cache de reponses (bug observe : « répète »
renvoyait la reponse a « donne un code python »)."""
from aura import filtre_instantane
from aura.regles_reponse import _RE_SUITE_CONVERSATION


def test_repete_n_est_pas_servi_par_le_cache_semantique():
    assert filtre_instantane.repondre_cache("répète") is None
    assert filtre_instantane.repondre_cache("répète la réponse précédente") is None


def test_suite_conversation_reconnue_sans_accents():
    assert _RE_SUITE_CONVERSATION.match("repete")
    assert _RE_SUITE_CONVERSATION.match("redis le")
    assert _RE_SUITE_CONVERSATION.match("encore une fois")
