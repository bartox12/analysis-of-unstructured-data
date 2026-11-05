# Task: compute the evidence for Problem 8 of Problem Set 1 (repo: analysis-of-unstructured-data)

## Context
- Problems 5–7 are implemented. Don't change their src code or their notebook cells.
- Put all Problem 8 content directly under the existing `## Problem 8: Comparing the embeddings` header (it's the last section).
- Same rules as before: logic in `list01/src/`, the notebook only imports and displays.
- Reuse what exists: `src.data.load_digits_features`, `src.embeddings.compute_embeddings` / `sensitivity_grid` / `EmbeddingResult`, and `src.plotting.plot_2d_embedding`.
- You do NOT write the discussion. Leave one markdown cell at the end containing only `### Discussion\n\n*(to be written)*`. The written answer will be done separately from your outputs.

## Problem statement
Compare the three embeddings from Problem 7: which one preserves global distances, which preserves local neighbourhoods, and why can t-SNE/UMAP distances not be read as similarities? Relate your answer to Problems 1–3.

## Step 1 - Dependencies
- scipy and scikit-learn should already be installed, verify it. Add anything missing with `uv add`, and report it.

## Step 2 - `src/structure_metrics.py` (new file)
All distances are Euclidean on float64, X is the raw 64-d digits features, Z a 2-D embedding.
1. `trustworthiness_score(X, Z, k=10) -> float`: wraps `sklearn.manifold.trustworthiness`. This is the local measure that penalises false neighbours.
2. `knn_preservation(X, Z, k=10) -> float`: mean Jaccard overlap between each point's k-NN set in X and in Z (exclude the point itself). Use `sklearn.neighbors.NearestNeighbors`.
3. `knn_accuracy_2d(Z, y, k=5, cv=5, random_state=0) -> float`: mean cross-validated accuracy of `KNeighborsClassifier` on Z, with stratified folds. This measures how well local class structure is kept.
4. `distance_correlation(X, Z) -> float`: Spearman ρ between `scipy.spatial.distance.pdist(X)` and `pdist(Z)` over all pairs. This is the global measure.
5. `centroid_distance_correlation(X, Z, y) -> float`: Spearman ρ between the 45 pairwise distances of the 10 class centroids in X and in Z. This tests whether gaps between clusters mean anything.
6. `structure_table(X, results: dict[str, EmbeddingResult], y, k=10) -> pandas.DataFrame`: one row per method with all five metrics above, sorted in the order PCA, t-SNE, UMAP.
7. `sensitivity_table(X, grid, y, k=10) -> pandas.DataFrame`: the same metrics for every embedding in the Problem 7 sensitivity grid (method, parameter value, metrics).
8. `seed_stability(X, y, seeds=(0, 1, 2), ...) -> pandas.DataFrame`: re-run t-SNE and UMAP with different seeds (default params otherwise). Report `centroid_distance_correlation` between the embeddings of different seeds (Z_seed_a vs Z_seed_b, as 45 centroid distances), plus each run's trustworthiness. PCA is deterministic, so a single row with ρ = 1 is enough.
9. `relative_contrast(X, n_queries=200, random_state=0) -> dict`: for random query points q taken from the data, compute (D_max − D_min)/D_min over the remaining points and return the mean and std. Write `relative_contrast_uniform(n, d, ...)`, which does the same for n uniform points in [0,1]^d with d = 64 and with d = 2.

## Step 3 - `src/plotting.py` (add functions, keep the existing ones)
- `plot_shepard_diagrams(X, results, n_pairs=5000, random_state=0) -> Figure`: 1×3 panels (PCA, t-SNE, UMAP). Each scatters a random sample of pairs: original distance on the x-axis, embedded distance on the y-axis, small and semi-transparent markers. Put the Spearman ρ from `distance_correlation` in each title. Normalise each axis to [0, 1] by its max so the panels are comparable, and say so in the axis labels.
- `plot_metric_bars(structure_df) -> Figure`: grouped bars, with methods on the x-axis and one bar per metric (all metrics lie in [0, 1]), plus a legend.

## Step 4 - Notebook cells 
1. Markdown: a short description of each metric (one line each) and which aspect it measures: **local** (trustworthiness, kNN preservation, kNN accuracy) or **global** (distance correlation, centroid correlation). State k = 10 and the seeds.
2. Code: rebuild or reuse `results` from Problem 7 (Problem 7's variables are in scope when the notebook runs top to bottom, reuse them instead of recomputing). Display `structure_table(...)`.
3. Code: `plot_metric_bars(...)`.
4. Code: `plot_shepard_diagrams(...)`.
5. Code: display `sensitivity_table(X, grid, y)`, reusing `grid` from Problem 7.
6. Code: display `seed_stability(...)`.
7. Code: display the relative-contrast results (digits 64-d, uniform 64-d, uniform 2-d) as a small table.
8. Markdown: `### Discussion\n\n*(to be written)*`. Nothing else.

## Step 5 - Verify
- Run the notebook from the repo root. It must run end-to-end with outputs saved.
- Confirm the source of every cell outside Problem 8 is unchanged.
- Report: all the tables (structure, sensitivity, seed stability, relative contrast), the total runtime of the Problem 8 cells, and any deviation from this plan.
- Do not git commit.