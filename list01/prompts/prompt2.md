# Task: implement Problem 6 of Problem Set 1 (repo: analysis-of-unstructured-data)

## Context
- Repo root contains `pyproject.toml` and `list01/`.
- `list01/list1.ipynb` already has problems 1–4 solved. Do not modify those cells. There are empty header cells `## Problem 5 …` to `## Problem 8 …`, put all Problem 6 content directly under the `## Problem 6` header, before the `## Problem 7` header.
- `list01/src/` is a package. All logic lives there, the notebook only imports and displays.
- `list01/data/` holds input files.
- I'm on Windows, so use `pathlib` everywhere.
- The relevant theory is already in the notebook (Problem 4): PCA = SVD of the centred X, λ_j = σ_j²/n (sklearn uses n−1, which doesn't change directions or ratios), EVR_j = σ_j²/Σσ_l², and reconstruction error with k components = Σ_{j>k} σ_j². The implementation should mirror this.

## Problem statement
Implement PCA from scratch via the SVD and reproduce `sklearn.decomposition.PCA` on the digits dataset (agree to 10⁻⁶ on the explained-variance ratios). Plot the cumulative explained variance and the 2-D projection coloured by class.

## Step 1 - Dependencies
- scikit-learn should already be in `pyproject.toml` from Problem 5. Verify this. No other new dependencies are expected. If you think one is needed, add it and report why. 

## Step 2 - `src/data.py` 
- `load_digits_features()`: `sklearn.datasets.load_digits()` returning `X` as float64 with shape (1797, 64) and `y` as int with shape (1797,).

## Step 3 - `src/pca.py` (from scratch: numpy only, no sklearn imports in this file)
1. `class PCA` mirroring sklearn's interface:
   - `__init__(self, n_components)`
   - `fit(X) -> self`: cast to float64. `mean_ = X.mean(axis=0)`, `Xc = X - mean_`. `U, S, Vt = np.linalg.svd(Xc, full_matrices=False)`. Apply the sign convention (below). Keep the top k = `n_components`.
   - Attributes: `mean_`, `components_` (k, p), `singular_values_` (k,), `explained_variance_ = S**2 / (n - 1)` (k,), `explained_variance_ratio_` (k,), `n_components_`, `n_samples_`.
   - Ratio denominator = total variance over all singular values, not just the kept k. Store the full spectrum, e.g. `self._all_singular_values`.
   - `transform(X) = (X - mean_) @ components_.T`, `fit_transform`, `inverse_transform(Z) = Z @ components_ + mean_`.
2. `_flip_signs(U, Vt)`: deterministic sign convention. For each component, make the entry of `Vt[j]` with the largest absolute value positive, and flip the matching column of `U` too. (Write it, don't import sklearn's `svd_flip`.)
3. `compare_with_sklearn(X, n_components=None, n_check_components=10)`: fits ours and `sklearn.decomposition.PCA(n_components, svd_solver="full")`, then returns one row per quantity with the max absolute difference:
   - `explained_variance_ratio_` (all components)
   - `explained_variance_`
   - `singular_values_`
   - `components_`, first `n_check_components` only, after aligning each sklearn component's sign to ours
   - `transform(X)` on those components
   - reconstruction `inverse_transform(transform(X))` with k = 10.

   Add a `passes` column with tolerance 1e-6. 

4. `theory_checks(X, ks=(2, 5, 10, 20)) -> pandas.DataFrame`: numerical confirmation of Problem 4, using our PCA only:
   - (a) EVR equals $\frac{\sigma_j^2}{\sum_{\ell}\sigma_\ell^2}$.
   - (b) The eigenvalues of the covariance matrix equal `explained_variance_`.
   - (c) For each $k$, the reconstruction error $\sum_i \|x_i-\hat{x}_i\|^2$ equals $\sum_{j>k}\sigma_j^2$.
   Report the absolute or relative error for each.

5. `n_components_for_variance`: the smallest k with cumulative EVR >= threshold.

## Step 4 - `src/plotting.py` (add functions, keep previous ones)
- `plot_cumulative_explained_variance`: step or line plot of the cumulative EVR against the number of components, with per component EVR as light bars on the same axis. Draw horizontal dashed lines at the thresholds and vertical lines at the k reaching each one.
- `plot_2d_embedding`: scatter of the first two columns coloured by class.
- `plot_components_as_images`: small grid showing the first n principal axes as 8×8 images with a heatmap centred at 0. 

## Step 5 - Notebook cells
1. Markdown: a short intro, assumptions
2. Code: `X, y = load_digits_features()`, display the shapes and dtypes.
3. Code: `pca = PCA().fit(X)`, display `compare_with_sklearn(X)`, and check that every row passes.
4. Code: display `theory_checks(X)`.
5. Code: cumulative explained variance plot, then print k for 80/90/95%.
6. Code: `Z = PCA(n_components=2).fit_transform(X)`, then `plot_2d_embedding(Z, y, ...)`.
7. Code: `plot_components_as_images(...)`.

## Step 5 — Verify
- Run the notebook from the repo root. It must run end-to-end with outputs saved.
- Report: the dependencies added, the summary table, and anything you had to deviate from in this plan and why.
- Do not git commit. 