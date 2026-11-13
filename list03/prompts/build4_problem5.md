# Task: implement Problem 5 of Problem Set 3 (repo: analysis-of-unstructured-data)

Build prompt 4 of 5 (after `build3_problem4`).

## Context
- `list03/list3.ipynb` has Problems 1–4 solved. Do not modify those cells. Put all Problem 5 content directly under the `## Problem 5 …` header, before the `## Problem 6 …` header.
- Same rules as before: logic in `list03/src/`, the notebook only imports and displays; `pathlib` everywhere; same `src/` style.
- Reuse what exists: `src.data.load_newsgroups` (with `categories=None` for all 20 groups), `src.preprocessing.tokenize`. Don't duplicate it.

## Problem statement
Train word2vec with gensim on the same corpus and separately load pretrained GloVe. For both, return the nearest neighbours of five query words and evaluate on an analogy set (e.g. king − man + woman).

**Corpus decision (state it in the notebook):** word2vec is trained on all 18,846 posts of 20 Newsgroups (all 20 groups, train + test), because the six-group training set of Problem 4 is too small to learn useful embeddings. The embeddings are never used to classify the Problem 4 test posts, so this does not leak into Problem 4.

## Step 1 - Dependencies
- `uv add gensim`. Check that `import gensim` works under `uv run`.

## Step 2 - `src/data.py` (add, keep the existing code)
1. `GLOVE_NAME = "glove-wiki-gigaword-100"`; `load_glove(name=GLOVE_NAME) -> KeyedVectors` via `gensim.downloader.load` (import it inside the function). Docstring: 400k lower-case words, Wikipedia 2014 + Gigaword 5 (6B tokens), 100 dimensions; the first call downloads 128 MB to `~/gensim-data`.
2. `SEMANTIC_SECTIONS`: the first five sections of `questions-words.txt` (`capital-common-countries`, `capital-world`, `currency`, `city-in-state`, `family`); the other nine are syntactic.
3. `load_analogy_questions(path=None) -> DataFrame`: the Google analogy test set (Mikolov et al., 2013) shipped with gensim (`gensim.test.utils.datapath("questions-words.txt")`). Lines starting with `:` are section names; every other line is `a b c d`, lower-cased. Columns `section`, `kind` (`semantic`/`syntactic`), `a`, `b`, `c`, `d`.

## Step 3 - `src/preprocessing.py` (add)
- `sentences_for_embeddings(texts) -> list[list[str]]`: `tokenize` each post, keep stop words, no stemming, drop empty posts. Each post is one "sentence", so the window never crosses posts. Docstring: word2vec needs function words as context and must output real words for queries and analogies.

## Step 4 - `src/embeddings.py` (new)
1. `W2V_PARAMS`: `vector_size=100` (matches GloVe), `window=5`, `min_count=5`, `sg=1`, `negative=5`, `ns_exponent=0.75`, `sample=1e-3`, `epochs=5`, `workers=1`. Comment: skip-gram with k = 5 negative samples is the SGNS objective of Problem 3; `workers=1` makes training deterministic for a fixed seed.
2. `train_word2vec(sentences, random_state=0, **kwargs) -> Word2Vec`: `Word2Vec(sentences, seed=random_state, **{**W2V_PARAMS, **kwargs})`; store the training time in `model.train_time_s_` (`time.perf_counter`).
3. `neighbours_table(models, words, topn=8) -> DataFrame`: for every model (`dict[str, KeyedVectors]`) and query word, `kv.most_similar(w, topn=topn)` formatted as `"neighbour (0.74), ..."`; `"(not in vocabulary)"` for unknown words. One row per query (index `query`), one column per model.
4. `RESTRICT_VOCAB = 300_000` (candidate answers restricted to the most frequent words, as in gensim's `evaluate_word_analogies`).
5. `covered_questions(questions, models, restrict_vocab=RESTRICT_VOCAB) -> DataFrame`: questions whose four words are all among the first `restrict_vocab` words of **every** model (original index kept).
6. `solve_analogies(kv, questions, restrict_vocab=RESTRICT_VOCAB, exclude_inputs=True, batch_size=128) -> np.ndarray`: vectorised 3CosAdd, the same rule as `most_similar(positive=[b, c], negative=[a])`: unit vectors, target `b − a + c`, candidates = the first `restrict_vocab` normalised vectors, cosine via one matrix product per batch, input words set to `-inf` when `exclude_inputs`, `argmax` → predicted word.
7. `analogy_summary(models, questions, restrict_vocab=RESTRICT_VOCAB) -> DataFrame`: one row per model (index `model`) with `coverage (all questions)` (share of all questions the model covers on its own) and `accuracy, semantic (shared)`, `accuracy, syntactic (shared)`, `accuracy, all (shared)`, computed only on the questions **all** models cover. Put the number of shared questions in `table.attrs["n_shared"]`. A private helper for the shared-subset accuracy is fine. Docstring: coverage and accuracy answer different questions; a small vocabulary cannot attempt many questions, so accuracies are compared on the shared subset only.
8. `analogy_examples(models, examples, topn=3) -> DataFrame`: for hand-picked `(a, b, c, d)` tuples, each model's top `topn` answers from `most_similar(positive=[b, c], negative=[a], topn=topn)`, comma-joined, the expected `d` marked with `" ✓"`. If an input word is unknown, write `"(<words> not in vocabulary)"` instead of a wrong answer; if only `d` is unknown, append `"(<d> not in vocabulary)"`. Index `analogy` (`"a : b :: c : ?"`), first column `expected`, then one column per model.

## Step 5 - Notebook cells (under `## Problem 5 …`)
1. Markdown `**Approach.**` with four bullets: **word2vec** (gensim skip-gram with negative sampling, the parameters above, seed 0 and one worker thread, trained on all 18,846 posts and why; tokens as in Problem 4 but stop words kept and no stemming, so queries and analogies use real words), **GloVe** (`glove-wiki-gigaword-100`: 100 dimensions, 400k words, about 6 billion tokens of Wikipedia and news), **Nearest neighbours** (cosine, five query words, one per family of newsgroups: mouse, hockey, space, god, car), **Analogies** (Google analogy set, 19,544 questions *a : b :: c : ?*, semantic and grammatical; answer = word closest to $b - a + c$ excluding the inputs; coverage reported per model, accuracy compared only on questions both models cover).
2. Code: imports; `sentences = sentences_for_embeddings(load_newsgroups("all", categories=None).texts)`; `w2v = train_word2vec(sentences, random_state=0)`; `glove = load_glove()`; `embeddings = {"word2vec (20NG)": w2v.wv, "GloVe (6B)": glove}`; two `print` lines: word2vec training tokens and vocabulary size, GloVe vocabulary size.
3. Code: `neighbours_table(embeddings, ("mouse", "hockey", "space", "god", "car"), topn=5)`.
4. Code: a tuple `examples` of seven analogies with a short comment on each, then `analogy_examples(embeddings, examples, topn=1)`:
   `man : king :: woman : queen` (gender), `france : paris :: germany : berlin` (country–capital), `big : bigger :: small : smaller` (comparative), `walk : walking :: swim : swimming` (present participle), `walk : walked :: run : ran` (irregular past tense), `hockey : nhl :: basketball : nba` (domain: sports leagues), `microsoft : windows :: apple : mac` (polysemous word).
5. Markdown, about 70–100 words, from the actual table: which relation types each model gets right, the notable differences (also where word2vec beats GloVe, if it does), at least one failure or ambiguity (e.g. the polysemous *windows*), and corpus size/domain as a plausible reason. Say the examples are hand-picked and the benchmark below measures the overall gap. Do not explain 3CosAdd again.
6. Code: `questions = load_analogy_questions()`; `summary = analogy_summary(embeddings, questions)`; print how many of all questions both models cover (`summary.attrs["n_shared"]`); display `summary.round(3)`.
7. Markdown `**Observations**`, about 100–150 words in three short paragraphs:
   - nearest neighbours: how strongly each embedding reflects its corpus (domain-specific senses such as *mouse* as a device vs. an animal, car-forum neighbours of *car*); one vector per word, so the dominant sense in the corpus wins;
   - benchmark: the accuracy gap on the shared questions with one or two numbers, and word2vec's low coverage and why;
   - caveat: corpus size (training tokens vs. ~6B) and domain are plausible reasons, but the comparison cannot separate them from the training method; vectors of rare words are unreliable.

## Step 6 - Verify
- Run the notebook from inside `list03/`. It must run end-to-end with outputs saved; Problem 1–4 cells unchanged.
- Reproducibility check in a scratch run (not in the notebook): two `train_word2vec(sentences, random_state=0)` calls give identical `wv.vectors` (`np.array_equal`). Report the result.
- Report the dependencies added, the neighbours, the analogy examples and the summary table, plus the word2vec training time, and any deviation from this plan.
- For reference, the original solution got: 3,611,480 training tokens and a 26,770-word word2vec vocabulary; 9,443 of 19,544 questions covered by both models; coverage 0.483 (word2vec) vs. 0.997 (GloVe); shared accuracy 0.089 vs. 0.608.
- Do not git commit.
