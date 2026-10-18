"""Helper code for Problem Set 1: vector representations, similarity & dimensionality reduction.

The notebook ``list1.ipynb`` only imports from this package and displays results.
"""

import warnings

try:
    from tqdm import TqdmWarning
except ImportError:  # tqdm is optional; it is pulled in by umap-learn.
    pass
else:
    # pooch (used by librosa) imports ``tqdm.auto``, which warns inside a Jupyter kernel
    # when ipywidgets is not installed. We never show download progress bars.
    warnings.filterwarnings("ignore", message="IProgress not found", category=TqdmWarning)
