# Task: implement Problem 7 of Problem Set 1 (repo: analysis-of-unstructured-data)

## Context
- Repo root contains `pyproject.toml` and `list01/`.
- `list01/list1.ipynb` already has problems 1–4 solved. Do not modify those cells. There are empty header cells `## Problem 5 …` to `## Problem 8 …`, put all Problem 7 content directly under the `## Problem 7` header, before the `## Problem 7` header.
- `list01/src/` is a package. All logic lives there, the notebook only imports and displays.
- `list01/data/` holds input files.
- I'm on Windows, so use `pathlib` everywhere.
- Reuse existing code, don't duplicate it:
  - `src.data.load_digits_features()` gives X (1797, 64) float64 and y.
  - `src.pca.PCA` is our from-scratch PCA. Use it for the PCA panel, not sklearn's.
  - `src.plotting.plot_2d_embedding(Z, y, title, ax=None)` was built to accept `ax`, for exactly this side-by-side figure.

## Problem statement
Run t-SNE and UMAP on the same features. Produce the three 2-D embeddings side by side (PCA, t-SNE, UMAP).

(Problem 8, done later, asks which embedding keeps global distances and which keeps local neighbourhoods, and why t-SNE/UMAP distances can't be read as similarities. Problem 7 should produce the material for that, but not the analysis itself.)

## Step 1 — Dependencies
- `uv add umap-learn`. Don't edit versions by hand or use pip.
- Check that `import umap` works under `uv run`.

## Step 2 - `src/embeddings.py`
1. `@dataclass EmbeddingResult`: `name: str`, `Z: np.ndarray` (n, 2), `params: dict`, `runtime_s: float`, `extra: dict`.
2. `embed_pca(X) -> EmbeddingResult`: our `PCA(n_components=2)`. `extra` = EVR of the two components.
3. `embed_tsne(X, perplexity=30.0, random_state=0, **kwargs) -> EmbeddingResult`: `sklearn.manifold.TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", max_iter=1000, random_state=random_state, **kwargs)`
4. `embed_umap(X, n_neighbors=15, min_dist=0.1, random_state=0, **kwargs) -> EmbeddingResult`: `umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=random_state, n_jobs=1, **kwargs)`. Write `warmup_umap(X, n=200)`, a tiny fit on a subset that triggers numba JIT compilation, and call it inside `embed_umap` before timing.
5. Time each method with `time.perf_counter()`, timing only the fit/transform.
6. `compute_embeddings(X, random_state=0, tsne_kwargs=None, umap_kwargs=None) -> dict[str, EmbeddingResult]`, with keys `"PCA"`, `"t-SNE"`, `"UMAP"` in that order.
7. `runtime_table(results) -> pandas.DataFrame`: method, key params, runtime_s, extra info.
8. `sensitivity_grid(X, random_state=0, perplexities=(5, 30, 100), n_neighbors=(5, 15, 100)) -> dict[str, list[EmbeddingResult]]`: t-SNE over the perplexities and UMAP over the `n_neighbors` values, with everything else at the defaults above.

## Step 3 - `src/plotting.py` (add functions, keep the existing ones)
- `plot_embeddings_side_by_side`: 1×3 subplots, each drawn with `plot_2d_embedding(..., ax=ax)`. Put the method, key parameter and runtime in each title. Show one shared legend for digits 0–9 (outside the panels or beneath them) instead of three. Keep tick labels on the PCA panel (its axes are PC coordinates). For t-SNE and UMAP, hide tick labels and add a small caption that the axes have no units. Use the same marker size and palette in every panel.
- `plot_sensitivity_grid`: 2×3 subplots, with t-SNE perplexities on row 1 and UMAP `n_neighbors` on row 2. Show the parameter value in each title, hide ticks, and use one shared legend.

## Step 4 - Notebook cells
1. Markdown: setup and assumptions
2. Code: load X, y; `results = compute_embeddings(X)`; display `runtime_table(results)`.
3. Code: `plot_embeddings_side_by_side(results, y)`.
4. Code: `grid = sensitivity_grid(X)`; `plot_sensitivity_grid(grid, y)`.
5. Markdown "Observations" (4–6 sentences, based on the actual figures):
   - how separated the 10 classes are in each embedding
   - which digit pairs overlap or touch, and in which method
   - how cluster sizes, gaps and the overall layout change across perplexity and `n_neighbors`
   - the runtime ordering

## Step 5 - Verify
- Run the notebook from the repo root. It must run end-to-end with outputs saved.
- Reproducibility check (outside the notebook, in a scratch run): calling `compute_embeddings(X)` twice with the same seed gives identical `Z` (`np.array_equal`) for PCA and t-SNE, and identical or `allclose` for UMAP. Report the result.
- Do not git commit.