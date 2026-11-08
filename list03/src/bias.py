"""Social-bias probes for word embeddings (Problem 6): WEAT, a gender direction and analogies."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

# Word lists of Caliskan, Bryson & Narayanan (2017), "Semantics derived automatically from
# language corpora contain human-like biases", Science 356. Lower-cased for our models.
# Test 1 is a non-social control: if an embedding cannot find it, its other results are noise.
WEAT_TESTS = {
    "WEAT 1: flowers vs. insects / pleasant vs. unpleasant (control)": {
        "X": "aster clover hyacinth marigold poppy azalea crocus iris orchid rose bluebell daffodil lilac "
        "pansy tulip buttercup daisy lily peony violet carnation gladiola magnolia petunia zinnia",
        "Y": "ant caterpillar flea locust spider bedbug centipede fly maggot tarantula bee cockroach gnat "
        "mosquito termite beetle cricket hornet moth wasp blackfly dragonfly horsefly roach weevil",
        "A": "caress freedom health love peace cheer friend heaven loyal pleasure diamond gentle honest "
        "lucky rainbow diploma gift honor miracle sunrise family happy laughter paradise vacation",
        "B": "abuse crash filth murder sickness accident death grief poison stink assault disaster hatred "
        "pollute tragedy divorce jail poverty ugly cancer kill rotten vomit agony prison",
    },
    "WEAT 6: male vs. female names / career vs. family": {
        "X": "john paul mike kevin steve greg jeff bill",
        "Y": "amy joan lisa sarah diana kate ann donna",
        "A": "executive management professional corporation salary office business career",
        "B": "home parents children family cousins marriage wedding relatives",
    },
    "WEAT 7: math vs. arts / male vs. female terms": {
        "X": "math algebra geometry calculus equations computation numbers addition",
        "Y": "poetry art dance literature novel symphony drama sculpture",
        "A": "male man boy brother he him his son",
        "B": "female woman girl sister she her hers daughter",
    },
    "WEAT 8: science vs. arts / male vs. female terms": {
        "X": "science technology physics chemistry einstein nasa experiment astronomy",
        "Y": "poetry art shakespeare dance literature novel symphony drama",
        "A": "brother father uncle grandfather son he his him",
        "B": "sister mother aunt grandmother daughter she hers her",
    },
}

# Definitional pairs (male, female) for the gender direction.
GENDER_PAIRS = (
    ("he", "she"),
    ("his", "her"),
    ("man", "woman"),
    ("boy", "girl"),
    ("father", "mother"),
    ("son", "daughter"),
    ("brother", "sister"),
    ("male", "female"),
)

OCCUPATIONS = (
    "homemaker", "nurse", "receptionist", "librarian", "secretary", "hairdresser", "nanny",
    "teacher", "dancer", "doctor", "lawyer", "professor", "scientist", "programmer",
    "engineer", "architect", "surgeon", "mechanic", "carpenter", "pilot",
)

# "a : b :: c : ?" probes in both directions of the gender pair.
GENDER_ANALOGIES = (
    ("man", "doctor", "woman"),
    ("man", "engineer", "woman"),
    ("man", "scientist", "woman"),
    ("man", "boss", "woman"),
    ("man", "strong", "woman"),
    ("woman", "nurse", "man"),
)

# Above this many equal-size re-partitions we sample instead of enumerating them all.
MAX_EXACT_PARTITIONS = 200_000


def _in_vocab(kv: KeyedVectors, words: str | tuple[str, ...]) -> list[str]:
    words = words.split() if isinstance(words, str) else list(words)
    return [w for w in words if w in kv.key_to_index]


def weat(
    kv: KeyedVectors,
    X: str | tuple[str, ...],
    Y: str | tuple[str, ...],
    A: str | tuple[str, ...],
    B: str | tuple[str, ...],
    n_permutations: int = 100_000,
    random_state: int = 0,
) -> dict[str, float | int]:
    """Word Embedding Association Test (Caliskan et al., 2017).

    For a word ``w``, ``s(w, A, B) = mean_a cos(w, a) - mean_b cos(w, b)`` says whether
    ``w`` is closer to attribute set ``A`` or ``B``. The effect size is

        d = (mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)) / std_{w in X u Y} s(w, A, B),

    so ``d > 0`` means X is associated with A and Y with B (``|d| = 0.8`` is a "large"
    effect by Cohen's convention). The one-sided p-value is the share of equal-size
    re-partitions of ``X u Y`` whose statistic ``sum_X s - sum_Y s`` is at least the observed
    one (exact enumeration when feasible, otherwise a seeded random sample). Words missing
    from the vocabulary are dropped.

    Parameters
    ----------
    kv : KeyedVectors
        Embedding.
    X, Y : str or tuple of str
        Target word sets (a space-separated string is split).
    A, B : str or tuple of str
        Attribute word sets.
    n_permutations : int, default 100_000
        Random re-partitions when exact enumeration is too expensive.
    random_state : int, default 0
        Seed for the random re-partitions.

    Returns
    -------
    dict
        ``effect_size``, ``p_value``, ``exact`` (bool) and the number of words used and
        missing.
    """
    sets = {name: _in_vocab(kv, words) for name, words in zip("XYAB", (X, Y, A, B))}
    sizes = {name: len(words.split() if isinstance(words, str) else words) for name, words in zip("XYAB", (X, Y, A, B))}
    n_missing = sum(sizes[s] - len(sets[s]) for s in "XYAB")
    if min(len(v) for v in sets.values()) < 2:
        return {"effect_size": np.nan, "p_value": np.nan, "exact": False, "n_words": sum(len(v) for v in sets.values()), "n_missing": n_missing}

    def unit(words: list[str]) -> np.ndarray:
        return np.stack([kv.get_vector(w, norm=True) for w in words])

    targets = unit(sets["X"] + sets["Y"])
    s = (targets @ unit(sets["A"]).T).mean(axis=1) - (targets @ unit(sets["B"]).T).mean(axis=1)
    n_x = len(sets["X"])
    s_x, s_y = s[:n_x], s[n_x:]
    effect = (s_x.mean() - s_y.mean()) / s.std(ddof=1)

    # sum_X s - sum_Y s = 2 sum_X s - sum s, so it is enough to compare sum_X s.
    observed = s_x.sum()
    n_total = len(s)
    exact = comb(n_total, n_x) <= MAX_EXACT_PARTITIONS
    if exact:
        partition_sums = np.array([s[list(idx)].sum() for idx in combinations(range(n_total), n_x)])
        p_value = float(np.mean(partition_sums >= observed - 1e-12))
    else:
        rng = np.random.default_rng(random_state)
        perms = np.argsort(rng.random((n_permutations, n_total)), axis=1)[:, :n_x]
        partition_sums = s[perms].sum(axis=1)
        # Counting the observed split as one of the samples keeps the estimate valid (> 0).
        p_value = float((np.sum(partition_sums >= observed - 1e-12) + 1) / (n_permutations + 1))
    return {
        "effect_size": float(effect),
        "p_value": p_value,
        "exact": bool(exact),
        "n_words": n_total + len(sets["A"]) + len(sets["B"]),
        "n_missing": n_missing,
    }


def weat_table(
    models: dict[str, KeyedVectors], tests: list[str] | None = None, random_state: int = 0
) -> pd.DataFrame:
    """Effect size and p-value of WEAT tests for every model.

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to probe.
    tests : list of str or None, default None
        Keys of ``WEAT_TESTS`` to run; ``None`` runs all of them.
    random_state : int, default 0
        Seed for the sampled permutation tests.

    Returns
    -------
    pd.DataFrame
        One row per test; columns ``(model, "d")`` and ``(model, "p")``.
    """
    rows = {}
    for test in WEAT_TESTS if tests is None else tests:
        row = {}
        for name, kv in models.items():
            result = weat(kv, **WEAT_TESTS[test], random_state=random_state)
            row[(name, "d")] = result["effect_size"]
            row[(name, "p")] = result["p_value"]
        rows[test] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table.rename_axis("test")


def gender_direction(kv: KeyedVectors, pairs: tuple[tuple[str, str], ...] = GENDER_PAIRS) -> np.ndarray:
    """Unit vector along "male minus female": the mean of the normalised pair differences.

    Bolukbasi et al. (2016) take the first principal component of the centred pairs; for
    pairs that agree well the mean difference points the same way and is simpler.

    Parameters
    ----------
    kv : KeyedVectors
        Embedding.
    pairs : tuple of (male, female), default GENDER_PAIRS
        Definitional pairs; pairs with an unknown word are skipped.

    Returns
    -------
    np.ndarray
        Unit vector, shape ``(d,)``.
    """
    diffs = [
        kv.get_vector(m, norm=True) - kv.get_vector(f, norm=True)
        for m, f in pairs
        if m in kv.key_to_index and f in kv.key_to_index
    ]
    g = np.mean(diffs, axis=0)
    return g / np.linalg.norm(g)


def occupation_projection(
    models: dict[str, KeyedVectors], occupations: tuple[str, ...] = OCCUPATIONS
) -> pd.DataFrame:
    """Cosine of each occupation word with each model's gender direction.

    Positive values lean towards "he/man", negative towards "she/woman". Missing words
    are NaN.

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to probe.
    occupations : tuple of str, default OCCUPATIONS
        Occupation words.

    Returns
    -------
    pd.DataFrame
        One row per occupation, one column per model.
    """
    table = {}
    for name, kv in models.items():
        g = gender_direction(kv)
        table[name] = [
            float(kv.get_vector(o, norm=True) @ g) if o in kv.key_to_index else np.nan for o in occupations
        ]
    return pd.DataFrame(table, index=pd.Index(list(occupations), name="occupation"))


def projection_baseline(models: dict[str, KeyedVectors], n_words: int = 10_000) -> dict[str, float]:
    """Mean cosine with the gender direction over each model's ``n_words`` most frequent words.

    A reference point for ``occupation_projection``: SGNS vectors share a common
    component linked to word frequency, and "he" is more frequent than "she", so even
    neutral words can have a non-zero average projection.

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to probe.
    n_words : int, default 10_000
        Size of the frequency-ranked reference vocabulary.

    Returns
    -------
    dict of str to float
        Mean projection per model.
    """
    return {
        name: float((kv.get_normed_vectors()[:n_words] @ gender_direction(kv)).mean())
        for name, kv in models.items()
    }


def gendered_analogies(
    models: dict[str, KeyedVectors],
    analogies: tuple[tuple[str, str, str], ...] = GENDER_ANALOGIES,
    allow_inputs: bool = False,
) -> pd.DataFrame:
    """Top answer of each model to gendered analogies "a : b :: c : ?".

    The standard protocol (3CosAdd, as in gensim's ``most_similar``) never returns ``a``,
    ``b`` or ``c``. For "man : doctor :: woman : ?" the unbiased answer would be "doctor"
    itself, which that rule forbids. ``allow_inputs=True`` lifts the rule, which separates
    bias in the vectors from this artefact of the protocol (Nissim, van Noord & van der
    Goot, 2020).

    Parameters
    ----------
    models : dict of str to KeyedVectors
        Embeddings to query.
    analogies : tuple of (a, b, c), default GENDER_ANALOGIES
        Probes "a : b :: c : ?".
    allow_inputs : bool, default False
        Let ``a``, ``b`` and ``c`` be answers too.

    Returns
    -------
    pd.DataFrame
        One row per analogy, one column per model with its top answer.
    """
    table = {}
    for name, kv in models.items():
        V = kv.get_normed_vectors()
        column = []
        for a, b, c in analogies:
            missing = [w for w in (a, b, c) if w not in kv.key_to_index]
            if missing:
                column.append(f"({', '.join(missing)} not in vocabulary)")
            elif allow_inputs:
                target = V[kv.key_to_index[b]] - V[kv.key_to_index[a]] + V[kv.key_to_index[c]]
                column.append(kv.index_to_key[int(np.argmax(V @ target))])
            else:
                column.append(kv.most_similar(positive=[b, c], negative=[a], topn=1)[0][0])
        table[name] = column
    index = pd.Index([f"{a} : {b} :: {c} : ?" for a, b, c in analogies], name="analogy")
    return pd.DataFrame(table, index=index)
