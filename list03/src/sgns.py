"""Skip-gram with negative sampling (Problems 3 and 5): gradient check and the shifted-PMI link."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models import Word2Vec
from scipy.special import expit, log_expit
from scipy.stats import pearsonr, spearmanr

GRADIENT_TOLERANCE = 1e-6


def sgns_objective(v_w: np.ndarray, v_c: np.ndarray, V_neg: np.ndarray) -> float:
    """SGNS objective of one (centre, context) pair with sampled negatives (to be maximised).

    ``log sigma(v_c . v_w) + sum_i log sigma(-v_{w_i} . v_w)``: the expectation over the
    noise distribution is replaced by the ``k`` drawn samples, as in training.

    Parameters
    ----------
    v_w : np.ndarray
        Centre-word vector, shape ``(d,)``.
    v_c : np.ndarray
        Context vector of the observed context word, shape ``(d,)``.
    V_neg : np.ndarray
        Context vectors of the ``k`` negative samples, shape ``(k, d)``.

    Returns
    -------
    float
        Value of the objective.
    """
    return float(log_expit(v_c @ v_w) + np.sum(log_expit(-(V_neg @ v_w))))


def sgns_gradients(
    v_w: np.ndarray, v_c: np.ndarray, V_neg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic gradients of ``sgns_objective`` derived in Problem 3.

    Returns
    -------
    grad_w : np.ndarray
        ``(1 - sigma(v_c . v_w)) v_c - sum_i sigma(v_{w_i} . v_w) v_{w_i}``, shape ``(d,)``.
    grad_c : np.ndarray
        ``(1 - sigma(v_c . v_w)) v_w``, shape ``(d,)``.
    grad_neg : np.ndarray
        Row ``i``: ``-sigma(v_{w_i} . v_w) v_w``, shape ``(k, d)``.
    """
    pos_err = 1.0 - expit(v_c @ v_w)  # = sigma(-v_c . v_w)
    neg_err = expit(V_neg @ v_w)  # shape (k,)
    grad_w = pos_err * v_c - neg_err @ V_neg
    grad_c = pos_err * v_w
    grad_neg = -neg_err[:, None] * v_w[None, :]
    return grad_w, grad_c, grad_neg


def _numerical_gradient(f, x: np.ndarray, eps: float) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        f_plus = f()
        x[i] = old - eps
        f_minus = f()
        x[i] = old
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def gradient_check(dim: int = 10, k: int = 5, eps: float = 1e-5, random_state: int = 0) -> pd.DataFrame:
    """Compare the analytic SGNS gradients with central finite differences.

    Parameters
    ----------
    dim : int, default 10
        Embedding dimension.
    k : int, default 5
        Number of negative samples.
    eps : float, default 1e-5
        Finite-difference step.
    random_state : int, default 0
        Seed of the random vectors.

    Returns
    -------
    pd.DataFrame
        One row per parameter with the relative error
        ``||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric||)`` and ``passes``.
    """
    rng = np.random.default_rng(random_state)
    v_w, v_c = rng.normal(size=dim), rng.normal(size=dim)
    V_neg = rng.normal(size=(k, dim))
    analytic = dict(zip(("v_w (centre)", "v_c (context)", "v_{w_i} (negatives)"), sgns_gradients(v_w, v_c, V_neg)))
    params = {"v_w (centre)": v_w, "v_c (context)": v_c, "v_{w_i} (negatives)": V_neg}

    rows = []
    for name, x in params.items():
        numeric = _numerical_gradient(lambda: sgns_objective(v_w, v_c, V_neg), x, eps)
        diff = np.linalg.norm(analytic[name] - numeric)
        rel = diff / (np.linalg.norm(analytic[name]) + np.linalg.norm(numeric))
        rows.append({"parameter": name, "shape": str(x.shape), "relative_error": rel})
    table = pd.DataFrame(rows).set_index("parameter")
    table["passes"] = table["relative_error"] < GRADIENT_TOLERANCE
    return table


def cooccurrence_counts(
    sentences: list[list[str]], key_to_index: dict[str, int], window: int
) -> sp.csr_matrix:
    """Expected (centre, context) pair counts of gensim's skip-gram, without subsampling.

    Words outside the model vocabulary are removed before windowing, as gensim does.
    gensim shrinks the window at every position to a uniform size in ``1..window``, so a
    context at distance ``delta`` is used with probability ``(window - delta + 1) / window``;
    each pair is counted with that weight, in both directions.

    Parameters
    ----------
    sentences : list of list of str
        Training sentences.
    key_to_index : dict of str to int
        Model vocabulary.
    window : int
        Maximum window size.

    Returns
    -------
    sp.csr_matrix
        Symmetric matrix ``#(w, c)``, shape ``(|V|, |V|)``.
    """
    ids, sent_ids = [], []
    for s, sentence in enumerate(sentences):
        kept = [key_to_index[w] for w in sentence if w in key_to_index]
        ids.extend(kept)
        sent_ids.extend([s] * len(kept))
    ids, sent_ids = np.asarray(ids), np.asarray(sent_ids)

    rows, cols, vals = [], [], []
    for delta in range(1, window + 1):
        same = sent_ids[:-delta] == sent_ids[delta:]
        left, right = ids[:-delta][same], ids[delta:][same]
        weight = np.full(left.size, (window - delta + 1) / window)
        rows += [left, right]
        cols += [right, left]
        vals += [weight, weight]
    n = len(key_to_index)
    counts = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return counts.tocsr()  # duplicate entries are summed


def shifted_pmi_check(
    model: Word2Vec, sentences: list[list[str]], n_words: int = 5000, min_pair_count: float = 20.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the trained dot products ``w . c`` with the shifted PMI predicted by Problem 3.

    gensim feeds the vector of the context-position word (``model.wv``) to predict the
    centre word through its output vector (``model.syn1neg``) and draws the negatives for
    that output slot. For this input ``u`` and output ``v`` the optimum of Problem 3 is

        u . v = log( #(u, v) / (k * #(u) * P_n(v)) ),

    which equals ``PMI(u, v) - log k`` when ``P_n`` is the unigram distribution. gensim
    uses ``P_n(v) ~ count(v)^0.75`` and down-samples frequent words; keeping word ``v``
    with probability ``p_v`` scales ``#(u, v)`` by ``p_u p_v``. We compute three targets:
    plain PMI, the shifted PMI of Levy & Goldberg and the shifted PMI with gensim's noise
    distribution and subsampling. Pairs are restricted to the ``n_words`` most frequent
    words and to ``#(u, v) >= min_pair_count``, where the counts are reliable.

    Parameters
    ----------
    model : Word2Vec
        Trained skip-gram model with negative sampling.
    sentences : list of list of str
        Its training sentences.
    n_words : int, default 5000
        Vocabulary prefix used for the pairs.
    min_pair_count : float, default 20.0
        Minimum expected co-occurrence count of a pair.

    Returns
    -------
    summary : pd.DataFrame
        One row per target: Pearson r, Spearman rho and mean residual ``u . v - target``.
    pairs : pd.DataFrame
        One row per pair: ``dot`` and the three targets (for plotting).
    """
    kv, k = model.wv, model.negative
    C = cooccurrence_counts(sentences, kv.key_to_index, model.window)
    counts = np.array([kv.get_vecattr(w, "count") for w in kv.index_to_key], dtype=np.float64)
    keep = np.array([kv.get_vecattr(w, "sample_int") for w in kv.index_to_key], dtype=np.float64) / (2**32 - 1)
    keep = np.minimum(keep, 1.0)
    noise = counts**model.ns_exponent
    noise /= noise.sum()

    row_total = np.asarray(C.sum(axis=1)).ravel()  # #(u) = sum_v #(u, v)
    row_total_sub = np.asarray(C @ keep).ravel()  # sum_v #(u, v) p_v (the p_u factor cancels)
    total = C.sum()

    sub = C[:n_words, :n_words].tocoo()
    mask = sub.data >= min_pair_count
    u, v, c_uv = sub.row[mask], sub.col[mask], sub.data[mask]

    pmi = np.log(c_uv * total / (row_total[u] * row_total[v]))
    sppmi = pmi - np.log(k)
    sppmi_gensim = np.log(c_uv * keep[v] / (k * row_total_sub[u] * noise[v]))
    dot = np.einsum("ij,ij->i", kv.vectors[u], model.syn1neg[v])

    pairs = pd.DataFrame(
        {
            "u": np.array(kv.index_to_key, dtype=object)[u],
            "v": np.array(kv.index_to_key, dtype=object)[v],
            "count": c_uv,
            "dot": dot,
            "PMI": pmi,
            "PMI - log k": sppmi,
            "PMI - log k (gensim noise + subsampling)": sppmi_gensim,
        }
    )
    rows = []
    for target in ("PMI", "PMI - log k", "PMI - log k (gensim noise + subsampling)"):
        rows.append(
            {
                "target": target,
                "pearson_r": pearsonr(dot, pairs[target]).statistic,
                "spearman_rho": spearmanr(dot, pairs[target]).statistic,
                "mean(dot - target)": float(np.mean(dot - pairs[target])),
            }
        )
    summary = pd.DataFrame(rows).set_index("target")
    summary.attrs["n_pairs"] = len(pairs)
    return summary, pairs
