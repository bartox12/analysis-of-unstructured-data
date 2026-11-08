"""TF-IDF (Problems 1 and 4): the vectoriser we use and checks of its formula."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

TOLERANCE = 1e-12

# Terms in a single training document are mostly typos, names and number-like strings.
MIN_DF = 2


def identity(doc: list[str]) -> list[str]:
    """Analyzer for already tokenised documents (module level, so pipelines can be cloned)."""
    return doc


def make_vectorizer(**kwargs) -> TfidfVectorizer:
    """TF-IDF vectoriser for pre-tokenised documents.

    ``sublinear_tf=True`` replaces the raw count by ``1 + ln(tf)``: the tenth occurrence of
    a word adds much less evidence than the first (Problem 1). Rows are L2-normalised, so
    long and short posts are comparable.

    Parameters
    ----------
    **kwargs
        Overrides for ``TfidfVectorizer``.

    Returns
    -------
    TfidfVectorizer
        Unfitted vectoriser.
    """
    params = {
        "analyzer": identity,
        "min_df": MIN_DF,
        "sublinear_tf": True,
        "smooth_idf": True,
        "norm": "l2",
        "dtype": np.float64,
        **kwargs,
    }
    return TfidfVectorizer(**params)


def course_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """Inverse document frequency as in Problem 1 and the lecture: ``ln(N / df)``."""
    return np.log(n_docs / np.asarray(df, dtype=np.float64))


def tfidf_from_counts(
    counts: sp.spmatrix, idf: np.ndarray, sublinear_tf: bool = True, norm: bool = True
) -> sp.csr_matrix:
    """Weight a document-term count matrix by ``tf * idf`` (from scratch).

    Parameters
    ----------
    counts : sparse matrix
        Raw term counts ``n_{t,d}``, shape ``(n_docs, n_terms)``.
    idf : np.ndarray
        One idf value per term, shape ``(n_terms,)``.
    sublinear_tf : bool, default True
        Use ``1 + ln(n_{t,d})`` instead of ``n_{t,d}`` for non-zero counts.
    norm : bool, default True
        Scale every row to unit L2 norm.

    Returns
    -------
    sp.csr_matrix
        TF-IDF weights, same shape as ``counts``.
    """
    tf = sp.csr_matrix(counts, dtype=np.float64, copy=True)
    if sublinear_tf:
        tf.data = 1.0 + np.log(tf.data)
    weights = tf @ sp.diags(idf)
    if norm:
        row_norms = np.sqrt(np.asarray(weights.multiply(weights).sum(axis=1)).ravel())
        row_norms[row_norms == 0] = 1.0
        weights = sp.diags(1.0 / row_norms) @ weights
    return sp.csr_matrix(weights)


def tfidf_checks(docs: list[list[str]], vectorizer: TfidfVectorizer) -> pd.DataFrame:
    """Check scikit-learn's TF-IDF against the formula of Problem 1.

    scikit-learn smooths the idf as if one extra document contained every term and adds 1,
    ``idf = ln((1 + N) / (1 + df)) + 1``, so a term in every document keeps a small weight.
    The check recomputes the idf and the whole matrix from raw counts.

    Parameters
    ----------
    docs : list of list of str
        Tokenised training documents.
    vectorizer : TfidfVectorizer
        Vectoriser fitted on ``docs`` with ``smooth_idf=True``.

    Returns
    -------
    pd.DataFrame
        One row per check with the maximum absolute difference and ``passes``.
    """
    vocabulary = vectorizer.vocabulary_
    counts = CountVectorizer(analyzer=identity, vocabulary=vocabulary).transform(docs)
    n_docs = counts.shape[0]
    df = np.bincount(counts.indices, minlength=len(vocabulary))

    idf = np.log((1 + n_docs) / (1 + df)) + 1
    weights = tfidf_from_counts(counts, idf, sublinear_tf=vectorizer.sublinear_tf)
    sklearn_weights = vectorizer.transform(docs)

    rows = [
        ("idf_ = ln((1+N)/(1+df)) + 1", np.max(np.abs(vectorizer.idf_ - idf))),
        ("W = l2norm((1 + ln tf) * idf)", abs(sklearn_weights - weights).max()),
    ]
    table = pd.DataFrame(rows, columns=["check", "max_abs_diff"]).set_index("check")
    table["passes"] = table["max_abs_diff"] < TOLERANCE
    return table


def idf_table(
    docs: list[list[str]], vectorizer: TfidfVectorizer, terms: tuple[str, ...] = (), n_lowest: int = 8
) -> pd.DataFrame:
    """Document frequency and idf of the most common terms and of a few chosen terms.

    Parameters
    ----------
    docs : list of list of str
        Tokenised training documents the vectoriser was fitted on.
    vectorizer : TfidfVectorizer
        Fitted vectoriser.
    terms : tuple of str, default ()
        Extra terms (e.g. topic words) to list after the ``n_lowest`` lowest-idf terms.
    n_lowest : int, default 8
        Number of lowest-idf terms.

    Returns
    -------
    pd.DataFrame
        Columns ``kind``, ``df``, ``df / N``, ``ln(N/df)`` (Problem 1) and ``idf_``
        (scikit-learn), indexed by term.
    """
    n_docs = len(docs)
    vocabulary = vectorizer.vocabulary_
    counts = CountVectorizer(analyzer=identity, vocabulary=vocabulary).transform(docs)
    df = np.bincount(counts.indices, minlength=len(vocabulary))
    names = vectorizer.get_feature_names_out()

    lowest = np.argsort(vectorizer.idf_, kind="stable")[:n_lowest]
    picks = [("most common", int(i)) for i in lowest] + [("chosen", vocabulary[t]) for t in terms]
    return pd.DataFrame(
        {
            "kind": [kind for kind, _ in picks],
            "df": [df[i] for _, i in picks],
            "df / N": [df[i] / n_docs for _, i in picks],
            "ln(N/df)": [course_idf(df[i], n_docs) for _, i in picks],
            "idf_": [vectorizer.idf_[i] for _, i in picks],
        },
        index=pd.Index([names[i] for _, i in picks], name="term"),
    )


def raw_vs_tfidf_terms(
    raw_tokens: list[str], doc: list[str], vectorizer: TfidfVectorizer, top_n: int = 10
) -> pd.DataFrame:
    """Top terms of one post ranked by raw count and by TF-IDF weight.

    Parameters
    ----------
    raw_tokens : list of str
        The post's lower-cased tokens with nothing removed (raw term frequency).
    doc : list of str
        The same post after our preprocessing (stop words removed, stemmed).
    vectorizer : TfidfVectorizer
        Fitted vectoriser.
    top_n : int, default 10
        Number of terms per ranking.

    Returns
    -------
    pd.DataFrame
        One row per rank: the term and its count, then the stem and its TF-IDF weight.
    """
    raw = pd.Series(raw_tokens).value_counts()
    raw = raw.reset_index().sort_values(["count", "index"], ascending=[False, True]).head(top_n)

    terms = vectorizer.get_feature_names_out()
    weights = vectorizer.transform([doc]).tocoo()
    order = np.lexsort((weights.col, -weights.data))[:top_n]  # ties broken alphabetically

    columns = pd.MultiIndex.from_tuples(
        [("raw count", "term"), ("raw count", "count"), ("TF-IDF", "stem"), ("TF-IDF", "weight")]
    )
    data = [
        [term, int(count), terms[weights.col[i]], float(weights.data[i])]
        for (term, count), i in zip(raw.itertuples(index=False), order)
    ]
    return pd.DataFrame(data, columns=columns, index=pd.RangeIndex(1, len(data) + 1, name="rank"))
