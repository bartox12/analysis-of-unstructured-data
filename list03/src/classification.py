"""TF-IDF text classification (Problem 4): Naive Bayes vs. logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .tfidf import make_vectorizer

MODEL_ORDER = ("Naive Bayes", "Logistic regression")

# alpha = 1 is Laplace smoothing; smaller values (Lidstone) trust the counts more.
PARAM_GRIDS = {
    "Naive Bayes": {"clf__alpha": [0.01, 0.03, 0.1, 0.3, 1.0]},
    "Logistic regression": {"clf__C": [1.0, 3.0, 10.0, 30.0, 100.0]},
}


def _make_pipeline(name: str) -> Pipeline:
    if name == "Naive Bayes":
        clf = MultinomialNB()
    elif name == "Logistic regression":
        # Multinomial (softmax) logistic regression with an L2 penalty, solved by L-BFGS.
        clf = LogisticRegression(max_iter=5000)
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("tfidf", make_vectorizer()), ("clf", clf)])


def fit_classifiers(
    docs: list[list[str]], y: np.ndarray, cv: int = 5, random_state: int = 0
) -> dict[str, GridSearchCV]:
    """Tune and fit both TF-IDF pipelines with stratified cross-validation on the training set.

    The vectoriser sits inside the pipeline, so in every fold the vocabulary and the idf are
    learnt from the training folds only. The selection metric is macro-F1, which weighs
    the six classes equally. Each search refits the best pipeline on the full training set.

    Parameters
    ----------
    docs : list of list of str
        Tokenised training documents.
    y : np.ndarray
        Training labels.
    cv : int, default 5
        Number of folds.
    random_state : int, default 0
        Seed of the fold shuffling.

    Returns
    -------
    dict of str to GridSearchCV
        Fitted searches, keyed by model name (``MODEL_ORDER``).
    """
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    searches = {}
    for name in MODEL_ORDER:
        search = GridSearchCV(_make_pipeline(name), PARAM_GRIDS[name], scoring="f1_macro", cv=folds)
        searches[name] = search.fit(docs, y)
    return searches


def cv_summary(searches: dict[str, GridSearchCV]) -> pd.DataFrame:
    """Selected hyper-parameter and its cross-validated macro-F1 (mean and std over folds).

    Parameters
    ----------
    searches : dict of str to GridSearchCV
        Output of ``fit_classifiers``.

    Returns
    -------
    pd.DataFrame
        One row per model.
    """
    rows = []
    for name, search in searches.items():
        (param, value), = search.best_params_.items()
        i = search.best_index_
        rows.append(
            {
                "model": name,
                "selected": f"{param.removeprefix('clf__')} = {value:g}",
                "cv_macro_f1": search.cv_results_["mean_test_score"][i],
                "cv_std": search.cv_results_["std_test_score"][i],
            }
        )
    return pd.DataFrame(rows).set_index("model")


def classification_table(
    models: dict[str, Pipeline], docs: list[list[str]], y: np.ndarray, target_names: list[str]
) -> pd.DataFrame:
    """Per-class precision, recall and F1 of every model on held-out documents.

    Parameters
    ----------
    models : dict of str to Pipeline
        Fitted pipelines.
    docs : list of list of str
        Tokenised test documents.
    y : np.ndarray
        Test labels.
    target_names : list of str
        Class names, indexed by label.

    Returns
    -------
    pd.DataFrame
        Rows: the classes, then ``macro avg`` and ``accuracy``. Columns: ``(model, metric)``
        plus the class ``support``.
    """
    labels = np.arange(len(target_names))
    blocks = {}
    for name, model in models.items():
        pred = model.predict(docs)
        p, r, f, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        block = pd.DataFrame({"precision": p, "recall": r, "f1": f}, index=list(target_names))
        block.loc["macro avg"] = [p.mean(), r.mean(), f.mean()]
        block.loc["accuracy"] = [np.nan, np.nan, np.mean(pred == y)]
        blocks[name] = block
    table = pd.concat(blocks, axis=1)
    table[("", "support")] = list(support) + [int(support.sum())] * 2
    return table


def confusion_matrices(
    models: dict[str, Pipeline], docs: list[list[str]], y: np.ndarray, n_classes: int
) -> dict[str, np.ndarray]:
    """Confusion matrix (rows = true class, columns = predicted class) of every model."""
    labels = np.arange(n_classes)
    return {name: confusion_matrix(y, model.predict(docs), labels=labels) for name, model in models.items()}


def linear_weights(model: Pipeline) -> np.ndarray:
    """Class weight vectors ``w_c`` of the linear decision rule, centred over the classes.

    For Naive Bayes ``w_c = log theta_{t|c}`` (Problem 2); for logistic regression the
    softmax weights. Softmax scores do not change when the same vector is added to every
    ``w_c``, so only the differences between classes matter. Subtracting the class mean
    makes the two models comparable: a large entry means "this term is more typical of
    class c than of the average class".

    Parameters
    ----------
    model : Pipeline
        Fitted ``tfidf`` + ``clf`` pipeline.

    Returns
    -------
    np.ndarray
        Centred weights, shape ``(K, |V|)``.
    """
    clf = model.named_steps["clf"]
    W = clf.feature_log_prob_ if isinstance(clf, MultinomialNB) else clf.coef_
    return W - W.mean(axis=0, keepdims=True)


def informative_features(
    models: dict[str, Pipeline], target_names: list[str], top_n: int = 10
) -> pd.DataFrame:
    """The ``top_n`` terms with the largest centred weight for each class and model.

    Parameters
    ----------
    models : dict of str to Pipeline
        Fitted pipelines.
    target_names : list of str
        Class names, indexed by label.
    top_n : int, default 10
        Terms per class.

    Returns
    -------
    pd.DataFrame
        One row per class, one column per model (comma-separated stems, strongest first).
    """
    table = {}
    for name, model in models.items():
        terms = model.named_steps["tfidf"].get_feature_names_out()
        W = linear_weights(model)
        table[name] = [", ".join(terms[np.argsort(-w, kind="stable")[:top_n]]) for w in W]
    return pd.DataFrame(table, index=pd.Index(list(target_names), name="class"))


def weight_agreement(models: dict[str, Pipeline], target_names: list[str]) -> pd.DataFrame:
    """Spearman correlation between the centred class weights of the two models.

    Both pipelines must share the vocabulary (same vectoriser settings and training data).

    Parameters
    ----------
    models : dict of str to Pipeline
        Exactly two fitted pipelines.
    target_names : list of str
        Class names, indexed by label.

    Returns
    -------
    pd.DataFrame
        One row per class with Spearman's rho over all terms.
    """
    (name_a, a), (name_b, b) = models.items()
    if a.named_steps["tfidf"].vocabulary_ != b.named_steps["tfidf"].vocabulary_:
        raise ValueError("the two pipelines use different vocabularies")
    W_a, W_b = linear_weights(a), linear_weights(b)
    rho = [spearmanr(wa, wb).statistic for wa, wb in zip(W_a, W_b)]
    return pd.DataFrame({"spearman_rho": rho}, index=pd.Index(list(target_names), name="class"))
