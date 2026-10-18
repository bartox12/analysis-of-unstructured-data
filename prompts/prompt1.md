# Task: Implement Problem 5 of Problem Set 1 (repo: analysis-of-unstructured-data)

## Context
- Repo root contains `pyproject.toml` and `list01/`.
- `list01/list1.ipynb` already has problems 1–4 solved. Do not modify those cells. There are empty header cells `## Problem 5 …` to `## Problem 8 …`, put all Problem 6 content directly under the `## Problem 6` header, before the `## Problem 7` header.
- `list01/src/` is a package. All logic lives there, the notebook only imports and displays. Relevant skeleton files: `src/modalities.py`, `src/plotting.py`.
- `list01/data/` holds input files.
- I'm on Windows, so use `pathlib` everywhere.

## Problem statement
Load one short text document, one image and one audio clip and turn each into a NumPy array. Report the shape and dtype of each, and write two sentences per modality explaining what a "feature vector" would mean in each case.

## Step 1 - Dependencies 
- Add only what Problem 5 needs

## Step 2 - `src/modalities.py`
Resolve the data dir as `DATA_DIR = Path(__file__).resolve().parents[1] / "data"`.

1. `fetch_samples` function, that takes data directory and overwrite boolean as parameters. It writes three files once (or skips if no `overwrite`), returns dictionary `{"text": ..., "image": ..., "audio": ...}`.
   - `text_sample.txt`:  short document from `sklearn.datasets.fetch_20newsgroups`. Take the first doc whose length is 400–1500 characters. Remove headers, footers and quotes. Save it as UTF-8.
   - `image_sample.png`: `skimage.data.chelsea()`.
   - `audio_sample.ogg`: copy the file at `librosa.ex("trumpet")`.
2. `load_text`: read UTF-8 text and return numpy 1-D array of UTF-8 bytes.
3. `load_image`: return `skimage.io.imread(path)`, keeping the original shape (H, W, 3) and dtype uint8.
4. `load_audio`: use `librosa.load`, return `(waveform, sr)`.
5. `describe_array`: returns modality, shape, ndim, dtype, size, nbytes, min, max, plus any extras (for audio: `sr` and `duration_s = n_samples / sr`, for text: `n_chars`).
6. `load_all` runs all of the individual `load...` functions and returns `dict[str, np.ndarray]` 
7. `summary_table` creates a dataframe with one row per modality built from `describe_array`.

## Step 3 - `src/plotting.py`
- `plot_modalities`: one figure with 3 panels: a text preview (first `n_text_chars` chars), the image (`imshow`, with its shape in the title), and the waveform vs time. Return the figure. don't show it.

## Step 4 - Notebook cells
1. Markdown: a short intro, assumptions
2. Code: imports from `src.modalities` and `src.plotting`, then `paths = fetch_samples()`.
3. Code: load the three arrays and display `summary_table(...)`.
5. Code: `plot_modalities(...)`.

## Step 5 — Verify
- Run the notebook from the repo root. It must run end-to-end with outputs saved.
- Report: the dependencies added, the summary table, and anything you had to deviate from in this plan and why.
- Do not git commit.