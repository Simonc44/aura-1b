"""Le routeur ne doit pas changer de NATURE en franchissant _SEUIL_SVM.

Defaut reel : sous 300 exemples, LogisticRegression -> probabilites calibrees ;
au-dessus, LinearSVC -> une softmax sur ses MARGES. Ces valeurs servent de
POIDS au mix multi-LoRA ({"math": 0,7, "web": 0,3}), donc franchir le seuil
changeait le comportement du modele pour la MEME question : la meme entree ne
produisait plus la meme ponderation d'adaptateurs.

Le correctif est l'echelle de Platt. Ces tests verrouillent la continuite :
les deux regimes rendent une probabilite, sur la meme echelle.
"""

from aura import routeur


def _jeu(n_par_classe: int = 130) -> tuple[list[str], list[str]]:
    """Jeu synthetique separable, >= _SEUIL_SVM exemples au total."""
    questions: list[str] = []
    labels: list[str] = []
    for i in range(n_par_classe):
        questions += [f"calcule la somme {i} plus {i}",
                      f"raconte l histoire ancienne numero {i}",
                      f"bonjour comment vas tu variante {i}"]
        labels += ["math", "web", "general"]
    return questions, labels


def test_seuil_franchi_avec_un_jeu_suffisant():
    questions, _ = _jeu()
    assert len(questions) >= routeur._SEUIL_SVM


def test_les_deux_regimes_rendent_une_probabilite(monkeypatch, tmp_path):
    """Sous le seuil ET au-dessus : dict qui somme a 1, classes completes."""
    questions, labels = _jeu()
    rendus = {}
    for coupe in (150, len(questions)):
        monkeypatch.setattr(routeur, "exemples_entrainement",
                            lambda q=questions[:coupe],
                            etiquettes=labels[:coupe]: (q, etiquettes))
        r = routeur.RouteurIntelligent(dossier_cache=tmp_path / f"c{coupe}")
        r._entrainer_classifieur()
        assert hasattr(r._classifieur, "predict_proba"), (
            f"a {coupe} exemples le classifieur n'a pas predict_proba : "
            f"les poids du mix LoRA ne sont plus des probabilites")
        X = r._vectoriseur.transform(["calcule 2 plus 2"])
        p = routeur._probas(r._classifieur, X)
        assert abs(sum(p.values()) - 1.0) < 1e-6, p
        assert set(p) == set(labels), p
        rendus[coupe] = p

    # Les deux regimes repondent la MEME chose sur une question evidente, et
    # avec des poids du meme ordre de grandeur (plus de softmax écrasée sur
    # les marges).
    for p in rendus.values():
        assert max(p.values()) >= 0.5, p
        assert p[max(p, key=lambda k: p[k])] == max(p.values())


def test_repli_softmax_si_pas_de_predict_proba():
    """Un classifieur sans predict_proba (cache ancien) reste utilisable."""
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer

    questions, labels = _jeu(40)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                          max_features=2000)
    X = vec.fit_transform(questions)
    brute = LinearSVC(C=1.0).fit(X, labels)
    assert not hasattr(brute, "predict_proba")
    p = routeur._probas(brute, vec.transform(["calcule 1 plus 1"]))
    assert abs(sum(p.values()) - 1.0) < 1e-6
    assert set(p) == set(labels)
