# README template for a submitted list

Fill in the `<...>` parts. Keep it short; the teacher should be able to run the
notebook in a few minutes by following it literally. Test every command before
submitting (see "Submission" in SKILL.md). Use `list01/README.md` as the worked example.

````markdown
# Analysis of Unstructured Data: Problem Set <N>

<Title of the problem set from the PDF>

Author: <name>

The solutions are in **`list<N>.ipynb`**, saved with all outputs, so it can be read
without running anything. Theory answers are written in Markdown; all code lives in
`src/`, and the notebook only calls it and discusses the results.

## Contents

| Problem | Type | Code |
|---|---|---|
| 1 | Theory | — (Markdown only) |
| 5 | Python | `src/<module>.py`, `src/plotting.py` |
| ... | ... | ... |

```
list0<N>/
├── list<N>.ipynb      solutions notebook
├── src/               implementation (one module per topic)
├── data/              small input files used by the notebook
├── prompts/           task prompts used while building the solution
├── pyproject.toml     dependencies
├── uv.lock            exact, locked versions
└── requirements.txt   same versions for pip
```

## Setup

Requirements: Python 3.12 and about <X> GB of disk space for the packages.

### Option A: uv (recommended)

Install uv once (https://docs.astral.sh/uv/getting-started/installation/), then in this folder:

```bash
uv sync
```

This creates `.venv/` with the exact versions from `uv.lock` (uv downloads Python 3.12 itself if it is missing).

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
uv run jupyter lab list<N>.ipynb
```

(With option B: activate `.venv` and run `jupyter lab list<N>.ipynb`.) Or open the folder in
VS Code / PyCharm and select the `.venv` interpreter as the kernel. Run all cells from the
top: later problems reuse variables from earlier ones.

To re-run everything from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute list<N>.ipynb --output list<N>_executed.ipynb
```

A full run takes about <T> on a laptop CPU. <Network / downloads note.>

## Notes

- <Seeds, determinism, which numbers vary between runs (runtimes).>
- <Anything the teacher should know.>
````
