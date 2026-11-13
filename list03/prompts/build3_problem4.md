# Task: implement Problem 4 of Problem Set 3 (repo: analysis-of-unstructured-data)

Build prompt 3 of 5 (after `build2_theory`).

## Context
- `list03/list3.ipynb` has Problems 1–3 answered. Do not modify those cells. Put all Problem 4 content directly under the `## Problem 4 …` header, before the `## Problem 5 …` header.
- `list03/src/` is a package. All logic lives there; the notebook only imports, calls and displays. A few plain lines (picking one example post, a `print`) can stay inline.
- `list03/data/` holds local input files. I'm on Windows, so use `pathlib` everywhere.
- Style of `src/` as in `list01/src/`: `from __future__ import annotations`, type hints, NumPy-style docstrings that explain *why* for non-obvious choices, UPPER_CASE module constants, pure functions, tables returned as `pandas.DataFrame`, plots returned as `Figure` (no `plt.show()` in `src/`).

## Problem statement
Build a TF-IDF pipeline (spaCy/NLTK preprocessing → `TfidfVectorizer`) and train both multinomial Naïve Bayes and logistic regression on a text-classification dataset (e.g. 20 Newsgroups subset). Report precision/recall/F1 and list the most informative features per class.

## Step 1 - Dependencies
- `uv add nltk scikit-learn`. Nothing else is expected; if you need more, add it with `uv add` and say why.

## Step 2 - `src/data.py`
- `DATA_DIR = Path(__file__).resolve().parents[1] / "data"`, `NLTK_DIR = DATA_DIR / "nltk_data"`.
- `CATEGORIES`: six newsgroups, two close pairs and two distant topics: `alt.atheism`, `soc.religion.christian`, `comp.graphics`, `sci.med`, `sci.space`, `rec.sport.hockey`.
- `REMOVE = ("headers", "footers", "quotes")`, with a comment: headers carry the newsgroup name and signatures identify regular posters, so a classifier could cheat.
- `@dataclass TextDataset`: `texts: list[str]`, `y: np.ndarray` (int64), `target_names: list[str]`.
- `load_newsgroups(subset, categories=CATEGORIES) -> TextDataset`: `fetch_20newsgroups(subset=..., categories=None if categories is None else list(categories), remove=REMOVE, shuffle=False)`. `subset` is `"train"`, `"test"` or `"all"`; `categories=None` keeps all 20 groups (needed in Problem 5).
- `english_stopwords() -> frozenset[str]`: NLTK's English stop words. Put `NLTK_DIR` first on `nltk.data.path`; on `LookupError` run `nltk.download("stopwords", download_dir=str(NLTK_DIR), quiet=True)` and retry. So the stop-word list lives in `data/nltk_data/` and ships with the repo.

## Step 3 - `src/preprocessing.py` (NLTK, as in the lecture)
- `TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?"` with an NLTK `RegexpTokenizer`: lower-case letters with an optional apostrophe part (don't, god's); numbers, e-mail fragments and punctuation are dropped. `MIN_TOKEN_LENGTH = 2`. Module-level `SnowballStemmer("english")`.
- `tokenize(text) -> list[str]`: lower-case, then tokenise.
- `preprocess_for_classification(text, stopwords) -> list[str]`: tokens not in `stopwords` and at least `MIN_TOKEN_LENGTH` long, Snowball-stemmed.
- `preprocess_corpus(data, stopwords=None) -> (docs, y, kept)`: preprocess every post (default stop words from `english_stopwords()`), drop posts left empty (they carry no evidence about the class), return the non-empty documents, their labels, and `kept`, the index of each kept post in `data.texts`.

## Step 4 - `src/tfidf.py`
- `MIN_DF = 2` (terms in a single training post are mostly typos, names and number-like strings).
- `identity(doc) -> doc`: analyzer for already tokenised documents. Module level, so pipelines can be cloned by `GridSearchCV`.
- `make_vectorizer(**kwargs) -> TfidfVectorizer`: `analyzer=identity`, `min_df=MIN_DF`, `sublinear_tf=True`, `smooth_idf=True`, `norm="l2"`, `dtype=np.float64`, overridable by `kwargs`. Docstring: sublinear tf because the tenth occurrence adds less evidence than the first (Problem 1); L2 rows make long and short posts comparable.
- `raw_vs_tfidf_terms(raw_tokens, doc, vectorizer, top_n=10) -> DataFrame`: the top `top_n` terms of one post by raw count (all lower-cased tokens, nothing removed; ties alphabetically) next to the top `top_n` stems by TF-IDF weight (`vectorizer.transform([doc])`; ties by column index). Columns as a MultiIndex `("raw count", "term")`, `("raw count", "count")`, `("TF-IDF", "stem")`, `("TF-IDF", "weight")`; index `rank` starting at 1.

## Step 5 - `src/classification.py`
1. `MODEL_ORDER = ("Naive Bayes", "Logistic regression")`. `PARAM_GRIDS`: NB `clf__alpha` in `[0.01, 0.03, 0.1, 0.3, 1.0]` (comment: 1 is Laplace, smaller is Lidstone); LR `clf__C` in `[1, 3, 10, 30, 100]`.
2. `_make_pipeline(name)`: `Pipeline([("tfidf", make_vectorizer()), ("clf", clf)])` with `MultinomialNB()` or `LogisticRegression(max_iter=5000)` (multinomial softmax, L2, L-BFGS).
3. `fit_classifiers(docs, y, cv=5, random_state=0) -> dict[str, GridSearchCV]`: `StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)`, `GridSearchCV(..., scoring="f1_macro", cv=folds)`, fitted on the training set only (default refit). Docstring: the vectoriser is inside the pipeline, so in every fold the vocabulary and idf come from the training folds only; macro-F1 weighs the six classes equally.
4. `cv_summary(searches) -> DataFrame`: one row per model, index `model`, columns `selected` (e.g. `"alpha = 0.03"`, formatted with `:g` and the `clf__` prefix removed), `cv_macro_f1`, `cv_std` (mean and std over folds at the best index). Nothing else (no search times).
5. `classification_table(models, docs, y, target_names) -> DataFrame`: per model a block of per-class `precision`, `recall`, `f1` (`precision_recall_fscore_support`, `zero_division=0`), plus rows `macro avg` and `accuracy` (accuracy in the `f1` column, NaN elsewhere). Concatenate the blocks with the model name as the top column level, and add a last column `("", "support")` with class supports and the total for the two summary rows.
6. `confusion_matrices(models, docs, y, n_classes) -> dict[str, np.ndarray]` (rows = true, columns = predicted).
7. `linear_weights(model) -> np.ndarray`: class weights of the linear decision rule, `feature_log_prob_` for NB (Problem 2) and `coef_` for LR, **centred over the classes** (subtract the mean over classes). Docstring: softmax scores do not change when the same vector is added to every class, so only differences matter, and centring makes the two models comparable.
8. `informative_features(models, target_names, top_n=10) -> DataFrame`: for each class and model, the `top_n` stems with the largest centred weight (stable sort), joined with `", "`. One row per class (index `class`), one column per model.

## Step 6 - `src/plotting.py`
- At the top, a small shared style: `SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")` (slot 1 = first model, slot 2 = second), a one-hue blue sequential colormap `BLUE_RAMP` (`LinearSegmentedColormap.from_list`, light `#f4f8fd` to dark `#0d366b`), text colours `INK = "#0b0b0b"`, `INK_SECONDARY = "#52514e"`, `INK_MUTED = "#898781"`, `GRID = "#e1e0d9"`, `BASELINE = "#c3c2b7"`, and a helper `_style_axes(ax, grid_axis="x")` (light grid behind the data, no top/right spines, muted tick labels). Later problems reuse it.
- `plot_confusion_matrices(cms, class_names) -> Figure`: one panel per model side by side (`constrained_layout`). Colour = row-normalised share (recall per true class) on one 0–1 scale shared by all panels with a single colour bar labelled "share of the true class"; every cell annotated with its raw count (white text on dark cells). Title `"<model> (accuracy xx.x%)"`, x labels rotated, y tick labels only on the first panel, no spines.

## Step 7 - Notebook cells (under `## Problem 4 …`)
1. Markdown `**Approach.**` with five short bullets: **Data** (the six groups, official train/test split, headers, signatures and quoted replies removed so the classifier cannot read the newsgroup name), **Preprocessing (NLTK)** (lower-cased alphabetic tokens, English stop words removed, Snowball stemming, empty posts dropped), **TF-IDF** (`min_df` 2, sublinear tf, L2 rows), **Classifiers** (MNB and LR, $\alpha$ and $C$ tuned by 5-fold CV on the training set with macro-F1; test set only for the final evaluation), **Informative features** (both models score classes linearly, so we list the terms whose weight favours the class most compared with the average class).
2. Code: imports from `src.classification`, `src.data`, `src.plotting`, `src.preprocessing`, `src.tfidf`; `train, test = load_newsgroups("train"), load_newsgroups("test")`; `docs_train, y_train, kept_train = preprocess_corpus(train)`; same for test; `class_names = train.target_names`; one `print` with the numbers of training and test posts and of empty posts dropped.
3. Code: `searches = fit_classifiers(docs_train, y_train, cv=5, random_state=0)`; `models = {name: s.best_estimator_ for name, s in searches.items()}`; display `cv_summary(searches)`.
4. Markdown: one line, "**Raw counts vs. TF-IDF.** The top terms of one `sci.space` training post, ranked by raw count (all tokens) and by TF-IDF weight (preprocessed stems):".
5. Code: take the fitted vectoriser `models["Naive Bayes"].named_steps["tfidf"]`; pick inline the first training post of `sci.space` with at least 150 tokens after preprocessing; display `raw_vs_tfidf_terms(tokenize(train.texts[kept_train[example]]), docs_train[example], vectorizer, top_n=8)`.
6. Code: `classification_table(models, docs_test, y_test, class_names).round(3)`.
7. Code: `fig = plot_confusion_matrices(confusion_matrices(models, docs_test, y_test, len(class_names)), class_names)`; `plt.show()`.
8. Code: `informative_features(models, class_names, top_n=8)`.
9. Markdown `**Observations**`, about 100–150 words in three short paragraphs, based on the actual outputs:
   - raw counts vs. TF-IDF in the example post (function words vs. specific terms);
   - which classifier is better and by how much, compared with the CV standard deviation (are they practically on par?); the easiest classes; the main confusion and why a bag of words causes it;
   - how the informative features differ between the models (general topic words vs. rare class-specific terms). Inspect the NB features of every class: if some come from one poster's e-mail signature that the footer removal missed, say so in two sentences (the model partly learns who wrote a post rather than its topic). Do not reproduce the signature.

## Step 8 - Verify
- Run the notebook from inside `list03/` (`uv run jupyter nbconvert --to notebook --execute --inplace list3.ipynb`). It must run end-to-end with outputs saved.
- Confirm the Problem 1–3 cells are unchanged.
- Report the dependencies added, the `cv_summary` and the macro-F1 / accuracy of both models, and any deviation from this plan.
- For reference, the original solution got: 3365 training and 2227 test posts (85 and 68 empty posts dropped); NB `alpha = 0.03` (CV macro-F1 ≈ 0.891), LR `C = 3` (≈ 0.877); test macro-F1 0.850 (NB) vs. 0.839 (LR), accuracy 0.858 vs. 0.846; main confusion `alt.atheism` → `soc.religion.christian`.
- Do not git commit.
