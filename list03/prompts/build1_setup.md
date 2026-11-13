# Task: set up the repo for Problem Set 3 (repo: analysis-of-unstructured-data)

Build prompt 1 of 5. Run them in order: `build1_setup` → `build2_theory` → `build3_problem4` → `build4_problem5` → `build5_problem6`.

## Context
- Repo root contains `materials/` (lecture PDFs and `AUD_Lab_assignments.pdf`) and one folder per problem set. `list01/` is the reference for structure and style.
- If the `aud-problem-set` skill is available, load it and follow it. In short: each list is a self-contained uv project, all logic lives in `listNN/src/`, and the notebook only imports, runs and discusses.
- I'm on Windows, so use `pathlib` everywhere.

## Problem statement
Problem Set 3, "Text I/II: classical representations, classification & embeddings" (`materials/AUD_Lab_assignments.pdf`): three [Theory] problems (TF-IDF, multinomial Naïve Bayes, skip-gram with negative sampling), two [Python] problems (TF-IDF classification, word2vec vs. GloVe) and one [Analysis] problem (social bias in embeddings).

## Step 1 - Project
Create `list03/` with:
- `pyproject.toml`: `name = "aud-list03"`, `requires-python = ">=3.12"`. Run `uv python pin 3.12`, then `uv add numpy pandas matplotlib ipykernel` and `uv add --dev nbconvert jupyterlab`. Never pip, never hand-edited versions.
- `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.DS_Store`.
- `data/.gitkeep`.
- `src/__init__.py` with only this docstring:
  `"""Helper code for Problem Set 3: text I/II: classical representations, classification & embeddings.\n\nThe notebook ``list3.ipynb`` only imports from this package and displays results.\n"""`

## Step 2 - Notebook skeleton
Create `list03/list3.ipynb` with `nbformat` (not hand-written JSON). Cells, all markdown:
1. Title cell:
   ```
   # Problem Set 3: Text I/II: classical representations, classification & embeddings

   Analysis of Unstructured Data. Author: <my name>.

   Problems 1–3 are theory answers in Markdown. Problems 4–6 call the code in `src/` and discuss the results. Run the cells from top to bottom: Problem 6 reuses the embeddings trained and loaded in Problem 5. The first run downloads 20 Newsgroups (14 MB, to `~/scikit_learn_data`) and GloVe (128 MB, to `~/gensim-data`).
   ```
2. `## Problem 1: TF-IDF weighting`
3. `## Problem 2: Multinomial Naïve Bayes`
4. `## Problem 3: Skip-gram with negative sampling`
5. `## Problem 4: TF-IDF classification with Naïve Bayes and logistic regression`
6. `## Problem 5: word2vec on 20 Newsgroups vs. pretrained GloVe`
7. `## Problem 6: Social bias in the embeddings`

Every later prompt puts its content directly under its own header, before the next header.

## Step 3 - Verify
- `uv sync` works from inside `list03/` and `uv run python -c "import src"` succeeds.
- Report the files created and the dependencies added.
- Do not git commit.
