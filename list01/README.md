# Analysis of Unstructured Data: Problem Set 1

Vector representations, similarity & dimensionality reduction

Author: Rafał Głodek

The solutions are in **`list1.ipynb`**, saved with all outputs, so it can be read
without running anything. Theory answers (Problems 1–4) are written in Markdown. For
Problems 5–8 all code lives in `src/`; the notebook only calls it, shows the results
and discusses them.

## Contents

| Problem | Type | Code |
|---|---|---|
| 1. Euclidean vs. cosine on the unit sphere | Theory | — |
| 2. Cosine distance is not a metric | Theory | — |
| 3. Curse of dimensionality | Theory | — |
| 4. PCA three ways | Theory | — |
| 5. Three modalities as NumPy arrays | Python | `src/modalities.py` |
| 6. PCA from scratch via the SVD | Python | `src/data.py`, `src/pca.py` |
| 7. PCA, t-SNE and UMAP embeddings | Python | `src/embeddings.py` |
| 8. Comparing the embeddings | Analysis | `src/structure_metrics.py` |

All figures come from `src/plotting.py`.

```
list01/
├── list1.ipynb        solutions notebook (with outputs)
├── src/               implementation, one module per topic
├── data/              text, image and audio samples used in Problem 5
├── prompts/           task prompts used while building the solution
├── pyproject.toml     dependencies
├── uv.lock            exact, locked versions
└── requirements.txt   the same versions for pip
```

## Setup

Requirements: Python 3.12 and about 700 MB of disk space for the packages.
No GPU and no internet access are needed after installing (all data is in `data/`
or ships with scikit-learn).

### Option A: uv (recommended)

Install uv once (https://docs.astral.sh/uv/getting-started/installation/), then run in this folder:

```bash
uv sync
```

This creates `.venv/` with the exact versions from `uv.lock`. If Python 3.12 is not
installed, uv downloads it automatically.

### Option B: pip

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the notebook

Open it in Jupyter Lab:

```bash
uv run jupyter lab list1.ipynb
```

With option B, activate `.venv` and run `jupyter lab list1.ipynb` instead. You can also
open the folder in VS Code or PyCharm and pick the `.venv` interpreter as the kernel.
Run the cells from top to bottom (**Run → Run All Cells**): Problem 8 reuses the
embeddings computed in Problem 7.

To execute the whole notebook from the command line without overwriting the submitted one:

```bash
uv run jupyter nbconvert --to notebook --execute list1.ipynb --output list1_executed.ipynb
```

A full run takes about 3 minutes on a laptop CPU (most of it in Problems 7 and 8).

## Notes

- All random steps use a fixed seed (`random_state=0`), so a re-run reproduces the
  tables and figures. Two exceptions: the runtimes in Problem 7 change from run to run,
  and t-SNE results can differ in the third decimal on another machine, because
  scikit-learn's multi-threaded t-SNE depends on the number of CPU threads. The
  conclusions do not change.
- The first UMAP call compiles code with numba, which adds a few seconds. This
  warm-up is excluded from the reported UMAP runtime.
- `data/` already contains the three Problem 5 samples. `fetch_samples()` only
  downloads them again (20 Newsgroups, librosa example) if the files are deleted.
