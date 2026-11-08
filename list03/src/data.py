"""Datasets shared across Problems 4-6: 20 Newsgroups, pretrained GloVe and the Google analogy set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from sklearn.datasets import fetch_20newsgroups

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NLTK_DIR = DATA_DIR / "nltk_data"

# Two close pairs (atheism vs. Christianity, space vs. medicine) and two distant topics.
CATEGORIES = (
    "alt.atheism",
    "soc.religion.christian",
    "comp.graphics",
    "sci.med",
    "sci.space",
    "rec.sport.hockey",
)

# Headers carry the newsgroup name and signatures identify regular posters; both would let a
# classifier "cheat", so we keep only the message body.
REMOVE = ("headers", "footers", "quotes")

GLOVE_NAME = "glove-wiki-gigaword-100"

# The first five sections of questions-words.txt are semantic, the remaining nine syntactic.
SEMANTIC_SECTIONS = (
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
)


@dataclass
class TextDataset:
    """Raw posts of one 20 Newsgroups split.

    Attributes
    ----------
    texts : list of str
        Message bodies (headers, signatures and quoted replies removed).
    y : np.ndarray
        Integer labels, shape ``(n,)``.
    target_names : list of str
        Newsgroup name of each label.
    """

    texts: list[str]
    y: np.ndarray
    target_names: list[str]


def load_newsgroups(subset: str, categories: tuple[str, ...] | None = CATEGORIES) -> TextDataset:
    """Load one split of 20 Newsgroups with headers, footers and quotes removed.

    The posts come in scikit-learn's fixed order (no shuffling), so results do not depend
    on a seed. The first call downloads about 14 MB to ``~/scikit_learn_data``.

    Parameters
    ----------
    subset : {"train", "test", "all"}
        Which split to load. ``"train"`` and ``"test"`` are the official date-based split.
    categories : tuple of str or None, default CATEGORIES
        Newsgroups to keep; ``None`` keeps all 20.

    Returns
    -------
    TextDataset
        The posts, labels and label names.
    """
    bunch = fetch_20newsgroups(
        subset=subset,
        categories=None if categories is None else list(categories),
        remove=REMOVE,
        shuffle=False,
    )
    return TextDataset(
        texts=list(bunch.data),
        y=np.asarray(bunch.target, dtype=np.int64),
        target_names=list(bunch.target_names),
    )


def english_stopwords() -> frozenset[str]:
    """Return NLTK's English stop-word list, downloading it into ``data/nltk_data`` once.

    Returns
    -------
    frozenset of str
        The 198 lower-case stop words (including forms such as ``"don't"``).
    """
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    try:
        words = nltk.corpus.stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", download_dir=str(NLTK_DIR), quiet=True)
        words = nltk.corpus.stopwords.words("english")
    return frozenset(words)


def load_glove(name: str = GLOVE_NAME) -> KeyedVectors:
    """Load pretrained GloVe vectors through ``gensim.downloader``.

    ``glove-wiki-gigaword-100`` holds 400k lower-case words trained on Wikipedia 2014 +
    Gigaword 5 (6B tokens), 100 dimensions. The first call downloads 128 MB to
    ``~/gensim-data``; parsing the text file takes about a minute.

    Parameters
    ----------
    name : str, default GLOVE_NAME
        Model name in the gensim-data catalogue.

    Returns
    -------
    KeyedVectors
        Word vectors ordered by corpus frequency.
    """
    import gensim.downloader as api

    return api.load(name)


def load_analogy_questions(path: Path | str | None = None) -> pd.DataFrame:
    """Read the Google analogy test set (Mikolov et al., 2013) shipped with gensim.

    Each line ``a b c d`` reads "a is to b as c is to d"; the model must recover ``d``
    from ``b - a + c``. Words are lower-cased because both our models are lower-case.

    Parameters
    ----------
    path : Path, str or None, default None
        Analogy file; ``None`` uses gensim's copy of ``questions-words.txt``.

    Returns
    -------
    pd.DataFrame
        One row per question, columns ``section``, ``kind`` (semantic/syntactic),
        ``a``, ``b``, ``c``, ``d``.
    """
    path = Path(datapath("questions-words.txt") if path is None else path)
    rows = []
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(":"):
            section = line[1:].strip()
            continue
        a, b, c, d = line.lower().split()
        rows.append((section, a, b, c, d))
    questions = pd.DataFrame(rows, columns=["section", "a", "b", "c", "d"])
    kind = np.where(questions["section"].isin(SEMANTIC_SECTIONS), "semantic", "syntactic")
    questions.insert(1, "kind", kind)
    return questions
