"""Word embeddings (Problem 5): word2vec on 20 Newsgroups vs. pretrained GloVe."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

# Skip-gram with k = 5 negative samples: the SGNS objective of Problem 3. 100 dimensions
# match the GloVe vectors. workers=1 makes training deterministic for a fixed seed (with
# several threads the order of the updates depends on thread scheduling).
W2V_PARAMS = {
    "vector_size": 100,
    "window": 5,
    "min_count": 5,
    "sg": 1,
    "negative": 5,
    "ns_exponent": 0.75,
    "sample": 1e-3,
    "epochs": 5,
    "workers": 1,
}

# Candidate answers are restricted to the most frequent words, as in gensim's
# ``evaluate_word_analogies`` (default 300k).
RESTRICT_VOCAB = 300_000


def train_word2vec(sentences: list[list[str]], random_state: int = 0, **kwargs) -> Word2Vec:
    """Train skip-gram with negative sampling on tokenised posts.

    Parameters
    ----------
    sentences : list of list of str
        Tokenised posts.
    random_state : int, default 0
        Seed for the initial vectors, the negative samples and the dynamic windows.
    **kwargs
        Overrides for ``W2V_PARAMS``.

    Returns
    -------
    Word2Vec
        Trained model; ``train_time_s_`` holds the training time. ``model.wv`` are the
        centre-word vectors, ``model.syn1neg`` the context ("output") vectors.
    """
    params = {**W2V_PARAMS, **kwargs}
    start = time.perf_counter()
    model = Word2Vec(sentences, seed=random_state, **params)
    model.train_time_s_ = time.perf_counter() - start
    return model


def vocabulary_summary(models: dict[str, KeyedVectors], words: tuple[str, ...] = ()) -> pd.DataFrame:
    """Vocabulary size and dimension of each model, and which ``words`` it lacks.

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to describe.
    words : tuple of str, default ()
        Words to look up.

    Returns
    -------
    pd.DataFrame
        One row per model.
    """
    rows = []
    for name, kv in models.items():
        missing = [w for w in words if w not in kv.key_to_index]
        rows.append(
            {
                "model": name,
                "vocabulary": len(kv),
                "dimension": kv.vector_size,
                "missing query words": ", ".join(missing) or "-",
            }
        )
    return pd.DataFrame(rows).set_index("model")


def neighbours_table(models: dict[str, KeyedVectors], words: tuple[str, ...], topn: int = 8) -> pd.DataFrame:
    """Nearest neighbours by cosine similarity of each query word in each model.

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to query.
    words : tuple of str
        Query words.
    topn : int, default 8
        Neighbours per word.

    Returns
    -------
    pd.DataFrame
        One row per query word, one column per model: ``"neighbour (cosine)"`` entries,
        nearest first.
    """
    table = {}
    for name, kv in models.items():
        column = []
        for w in words:
            if w not in kv.key_to_index:
                column.append("(not in vocabulary)")
                continue
            column.append(", ".join(f"{n} ({s:.2f})" for n, s in kv.most_similar(w, topn=topn)))
        table[name] = column
    return pd.DataFrame(table, index=pd.Index(list(words), name="query"))


def _vocab(kv: KeyedVectors, restrict_vocab: int) -> dict[str, int]:
    return {w: i for i, w in enumerate(kv.index_to_key[:restrict_vocab])}


def covered_questions(
    questions: pd.DataFrame, models: dict[str, KeyedVectors], restrict_vocab: int = RESTRICT_VOCAB
) -> pd.DataFrame:
    """Questions whose four words are all in the (restricted) vocabulary of every model.

    Parameters
    ----------
    questions : pd.DataFrame
        Analogy questions with columns ``a``, ``b``, ``c``, ``d``.
    models : dict of str to KeyedVectors
        Every model must know all four words.
    restrict_vocab : int, default RESTRICT_VOCAB
        Only the most frequent ``restrict_vocab`` words of each model count.

    Returns
    -------
    pd.DataFrame
        The covered subset (original index kept).
    """
    mask = np.ones(len(questions), dtype=bool)
    for kv in models.values():
        vocab = _vocab(kv, restrict_vocab)
        for col in "abcd":
            mask &= questions[col].isin(vocab).to_numpy()
    return questions[mask]


def solve_analogies(
    kv: KeyedVectors,
    questions: pd.DataFrame,
    restrict_vocab: int = RESTRICT_VOCAB,
    exclude_inputs: bool = True,
    batch_size: int = 128,
) -> np.ndarray:
    """Answer "a : b :: c : ?" by 3CosAdd: the word closest in cosine to ``b - a + c``.

    Vectorised version of gensim's ``most_similar(positive=[b, c], negative=[a])``: the
    unit-normalised vectors of ``b`` and ``c`` are added, that of ``a`` subtracted, and
    the candidate with the largest cosine wins. Questions must be covered by ``kv``.

    Parameters
    ----------
    kv : KeyedVectors
        Embedding.
    questions : pd.DataFrame
        Columns ``a``, ``b``, ``c``; every word must be among the first ``restrict_vocab``.
    restrict_vocab : int, default RESTRICT_VOCAB
        Candidates are the ``restrict_vocab`` most frequent words.
    exclude_inputs : bool, default True
        Remove ``a``, ``b`` and ``c`` from the candidates (the standard protocol).
    batch_size : int, default 128
        Questions per matrix product (memory ``batch_size * restrict_vocab`` floats).

    Returns
    -------
    np.ndarray
        Predicted word for every question, shape ``(len(questions),)``.
    """
    vocab = _vocab(kv, restrict_vocab)
    words = np.array(kv.index_to_key[:restrict_vocab], dtype=object)
    V = kv.get_normed_vectors()[: len(words)]
    idx = {col: questions[col].map(vocab).to_numpy() for col in "abc"}
    predictions = np.empty(len(questions), dtype=object)
    for start in range(0, len(questions), batch_size):
        sl = slice(start, start + batch_size)
        target = V[idx["b"][sl]] - V[idx["a"][sl]] + V[idx["c"][sl]]
        sims = target @ V.T
        if exclude_inputs:
            rows = np.arange(sims.shape[0])
            for col in "abc":
                sims[rows, idx[col][sl]] = -np.inf
        predictions[sl] = words[np.argmax(sims, axis=1)]
    return predictions


def compare_with_gensim(
    kv: KeyedVectors, questions: pd.DataFrame, n: int = 200, random_state: int = 0,
    restrict_vocab: int = RESTRICT_VOCAB,
) -> pd.DataFrame:
    """Check ``solve_analogies`` against gensim's ``most_similar`` on a random sample.

    Parameters
    ----------
    kv : KeyedVectors
        Embedding.
    questions : pd.DataFrame
        Covered questions.
    n : int, default 200
        Sample size.
    random_state : int, default 0
        Seed of the sample.
    restrict_vocab : int, default RESTRICT_VOCAB
        Candidate vocabulary size.

    Returns
    -------
    pd.DataFrame
        One row: sample size, number of identical answers and ``passes``.
    """
    sample = questions.sample(n=min(n, len(questions)), random_state=random_state)
    ours = solve_analogies(kv, sample, restrict_vocab=restrict_vocab)
    ref = [
        kv.most_similar(positive=[b, c], negative=[a], topn=1, restrict_vocab=restrict_vocab)[0][0]
        for a, b, c in sample[["a", "b", "c"]].itertuples(index=False)
    ]
    agree = int(np.sum(ours == np.array(ref, dtype=object)))
    return pd.DataFrame({"n_questions": [len(sample)], "same_answer": [agree], "passes": [agree == len(sample)]})


def analogy_accuracy(
    models: dict[str, KeyedVectors], questions: pd.DataFrame, restrict_vocab: int = RESTRICT_VOCAB
) -> pd.DataFrame:
    """Accuracy of every model on the analogy questions that all models cover.

    Restricting to the common subset makes the comparison fair: word2vec on 20 Newsgroups
    knows far fewer words than GloVe, and a question with an unknown word cannot be
    answered at all.

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to evaluate.
    questions : pd.DataFrame
        All analogy questions (``load_analogy_questions``).
    restrict_vocab : int, default RESTRICT_VOCAB
        Candidate vocabulary size.

    Returns
    -------
    pd.DataFrame
        Rows: the 14 sections, then the semantic, syntactic and overall totals. Columns:
        number of questions, number covered by all models and per-model accuracy.
    """
    common = covered_questions(questions, models, restrict_vocab).copy()
    for name, kv in models.items():
        common[name] = solve_analogies(kv, common, restrict_vocab) == common["d"].to_numpy()

    sections = list(dict.fromkeys(questions["section"]))
    kinds = questions.drop_duplicates("section").set_index("section")["kind"]

    def summarise(total: pd.DataFrame, covered: pd.DataFrame) -> dict[str, float]:
        row = {"kind": "", "n_questions": len(total), "n_common": len(covered)}
        for name in models:
            row[name] = covered[name].mean() if len(covered) else np.nan
        return row

    rows = {}
    for s in sections:
        rows[s] = summarise(questions[questions["section"] == s], common[common["section"] == s])
        rows[s]["kind"] = kinds[s]
    for kind in ("semantic", "syntactic"):
        rows[f"all {kind}"] = summarise(questions[questions["kind"] == kind], common[common["kind"] == kind])
        rows[f"all {kind}"]["kind"] = "total"
    rows["all"] = summarise(questions, common)
    rows["all"]["kind"] = "total"
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("section")


def own_coverage_accuracy(
    models: dict[str, KeyedVectors], questions: pd.DataFrame, restrict_vocab: int = RESTRICT_VOCAB
) -> pd.DataFrame:
    """Coverage and accuracy of each model on all the questions it can answer (gensim's protocol).

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to evaluate.
    questions : pd.DataFrame
        All analogy questions.
    restrict_vocab : int, default RESTRICT_VOCAB
        Candidate vocabulary size.

    Returns
    -------
    pd.DataFrame
        One row per model: covered questions, coverage and accuracy (semantic, syntactic, all).
    """
    rows = []
    for name, kv in models.items():
        covered = covered_questions(questions, {name: kv}, restrict_vocab)
        correct = solve_analogies(kv, covered, restrict_vocab) == covered["d"].to_numpy()
        rows.append(
            {
                "model": name,
                "n_covered": len(covered),
                "coverage": len(covered) / len(questions),
                "acc_semantic": correct[covered["kind"].to_numpy() == "semantic"].mean(),
                "acc_syntactic": correct[covered["kind"].to_numpy() == "syntactic"].mean(),
                "acc_all": correct.mean(),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def analogy_examples(
    models: dict[str, KeyedVectors], examples: tuple[tuple[str, str, str, str], ...], topn: int = 3
) -> pd.DataFrame:
    """Top answers of every model for a few hand-picked analogies "a : b :: c : d".

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to query.
    examples : tuple of (a, b, c, d)
        Analogies; ``d`` is the expected answer.
    topn : int, default 3
        Answers to show.

    Returns
    -------
    pd.DataFrame
        One row per analogy: the expected answer and, per model, the ``topn`` answers
        (best first, the expected one marked with a check mark). Unknown words are
        reported as such instead of counting as a wrong answer.
    """
    rows = []
    for a, b, c, d in examples:
        row = {"analogy": f"{a} : {b} :: {c} : ?", "expected": d}
        for name, kv in models.items():
            missing = [w for w in (a, b, c) if w not in kv.key_to_index]
            if missing:
                row[name] = f"({', '.join(missing)} not in vocabulary)"
                continue
            answers = [w for w, _ in kv.most_similar(positive=[b, c], negative=[a], topn=topn)]
            row[name] = ", ".join(f"{w} ✓" if w == d else w for w in answers)
            if d not in kv.key_to_index:
                row[name] += f" ({d} not in vocabulary)"
        rows.append(row)
    return pd.DataFrame(rows).set_index("analogy")


def analogy_summary(
    models: dict[str, KeyedVectors], questions: pd.DataFrame, restrict_vocab: int = RESTRICT_VOCAB
) -> pd.DataFrame:
    """Coverage of the whole test set and accuracy on the questions all models cover.

    Coverage and accuracy answer different questions: a model with a small vocabulary
    simply cannot attempt many questions, so accuracies are compared only on the shared
    subset (``analogy_accuracy``).

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to evaluate.
    questions : pd.DataFrame
        All analogy questions.
    restrict_vocab : int, default RESTRICT_VOCAB
        Candidate vocabulary size.

    Returns
    -------
    pd.DataFrame
        One row per model: share of all questions it can attempt, and accuracy on the
        shared questions (semantic, syntactic, all). ``attrs["n_shared"]`` holds the
        number of shared questions.
    """
    shared = analogy_accuracy(models, questions, restrict_vocab)
    rows = []
    for name, kv in models.items():
        n_covered = len(covered_questions(questions, {name: kv}, restrict_vocab))
        rows.append(
            {
                "model": name,
                "coverage (all questions)": n_covered / len(questions),
                "accuracy, semantic (shared)": shared.loc["all semantic", name],
                "accuracy, syntactic (shared)": shared.loc["all syntactic", name],
                "accuracy, all (shared)": shared.loc["all", name],
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    table.attrs["n_shared"] = int(shared.loc["all", "n_common"])
    return table
