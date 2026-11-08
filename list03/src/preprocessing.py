"""NLTK preprocessing (Problems 4-5): tokenisation, stop-word removal and stemming."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from .data import TextDataset, english_stopwords

# Lower-case letters with an optional apostrophe part ("don't", "god's"). Numbers, e-mail
# fragments and punctuation are dropped: they rarely carry topic information in these posts.
TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?"
MIN_TOKEN_LENGTH = 2

_TOKENIZER = RegexpTokenizer(TOKEN_PATTERN)
_STEMMER = SnowballStemmer("english")


def tokenize(text: str) -> list[str]:
    """Lower-case ``text`` and split it into alphabetic word tokens.

    Parameters
    ----------
    text : str
        Raw message body.

    Returns
    -------
    list of str
        Tokens in document order.
    """
    return _TOKENIZER.tokenize(text.lower())


def preprocess_for_classification(text: str, stopwords: frozenset[str]) -> list[str]:
    """Tokens for the TF-IDF classifiers: no stop words, Snowball stems.

    Stemming merges inflections ("galaxy"/"galaxies" -> "galaxi"), which helps the bag of
    words, where every surface form is a separate dimension.

    Parameters
    ----------
    text : str
        Raw message body.
    stopwords : frozenset of str
        Words to drop before stemming.

    Returns
    -------
    list of str
        Stems of the remaining tokens.
    """
    return [
        _STEMMER.stem(tok)
        for tok in tokenize(text)
        if tok not in stopwords and len(tok) >= MIN_TOKEN_LENGTH
    ]


def preprocess_corpus(
    data: TextDataset, stopwords: frozenset[str] | None = None
) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    """Preprocess every post of a split for classification and drop the empty ones.

    About 3% of the posts have no token left once headers, footers, quotes and stop words
    are removed. They contain no evidence about the class, so we leave them out of
    training and evaluation.

    Parameters
    ----------
    data : TextDataset
        Posts and labels.
    stopwords : frozenset of str or None, default None
        Stop words; ``None`` uses NLTK's English list.

    Returns
    -------
    docs : list of list of str
        Preprocessed, non-empty documents.
    y : np.ndarray
        Their labels.
    kept : np.ndarray
        Index of each kept document in ``data.texts``.
    """
    stopwords = english_stopwords() if stopwords is None else stopwords
    docs = [preprocess_for_classification(t, stopwords) for t in data.texts]
    kept = np.flatnonzero([len(d) > 0 for d in docs])
    return [docs[i] for i in kept], data.y[kept], kept


def sentences_for_embeddings(texts: Iterable[str]) -> list[list[str]]:
    """Token lists for word2vec: lower-case words, stop words kept, no stemming.

    word2vec needs function words as context and must output real words, so that queries
    like "king" and the analogy set can be looked up. Each post is one "sentence", so the
    context window never crosses posts.

    Parameters
    ----------
    texts : iterable of str
        Raw message bodies.

    Returns
    -------
    list of list of str
        One non-empty token list per post.
    """
    sentences = (tokenize(t) for t in texts)
    return [s for s in sentences if s]
