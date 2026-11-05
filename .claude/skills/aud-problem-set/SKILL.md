---
name: aud-problem-set
description: Solve a problem set of the "Analysis of Unstructured Data" course (list01, list02, ...) in the list01 convention - self-contained uv project per list, business logic in src/, a presentation notebook listN.ipynb with executed outputs and written answers. Use when the user asks to scaffold, solve, extend, verify, polish or prepare for submission any list / problem set / problem of this course.
argument-hint: "<listNN> [problem numbers] [extra constraints]"
---

# AUD problem sets (list01 convention)

`list01/` is the reference implementation. When unsure how something should look,
open `list01/` and copy its style: `list01/list1.ipynb`, `list01/src/*.py`,
`list01/prompts/*.md`. This skill overrides any generic "problem set" skill (no
`tests/`, no `src/listNN/` package, no ruff requirement here).

## Inputs

- Problem statements: `materials/AUD_Lab_assignments.pdf` (all sets; Problem Set N = `listNN`).
  Read with `pdftotext -layout`. Formulas extract badly, so reconstruct them carefully
  and check against the lecture slides.
- Theory: lecture PDFs in `materials/` (`1_Introduction...pdf`, `2_Data_Acquisition.pdf`, ...).
  Use their notation, terminology and method choices so answers look like they belong to the course.
- Course rules (top of the assignments PDF): every problem is tagged **[Theory]**, **[Python]**
  or **[Analysis]**; one runnable `.ipynb` per sheet; text answers in Markdown; every figure
  reproducible from the notebook; **state every assumption**; small public datasets; CPU-runnable.
- The user often hands over a detailed per-problem prompt (see `list01/prompts/`). Treat its
  function names, signatures, parameters and notebook cell list as requirements. Save it
  verbatim as the next `listNN/prompts/promptK.md` unless it is already there.

## Repo layout (one self-contained uv project per list)

```
listNN/
├── pyproject.toml     # name = "aud-listNN", requires-python = ">=3.12"; dev group: nbconvert, jupyterlab
├── uv.lock            # committed, exact environment
├── .python-version    # 3.12
├── .gitignore         # .venv/ __pycache__/ *.pyc .ipynb_checkpoints/ .DS_Store
├── README.md          # for the teacher: setup + how to run (see references/readme-template.md)
├── listN.ipynb        # the solutions, N without leading zero (list1.ipynb, list2.ipynb)
├── data/              # .gitkeep + small input files the notebook reads (only if needed)
├── prompts/           # task prompts used to build the list
└── src/
    ├── __init__.py    # module docstring: "Helper code for Problem Set N: <topic>. The notebook only imports..."
    ├── data.py        # dataset loaders shared across problems
    ├── <topic>.py     # one module per topic/algorithm (pca.py, embeddings.py, ...), not per problem
    └── plotting.py    # every non-trivial plot of the list
```

- `src` is a plain package (no build system, not installed). The notebook sits next to it,
  so `from src.pca import PCA` works because Jupyter's cwd is the notebook folder.
- Scaffold a new list: create the folder and files above, then from inside `listNN/`:
  `uv init --bare --name aud-listNN --python 3.12` (or write `pyproject.toml` by hand),
  `uv python pin 3.12`, `uv add numpy pandas matplotlib ipykernel`,
  `uv add --dev nbconvert jupyterlab`. Create the notebook with `nbformat` with a title
  cell and one `## Problem K: <short title>` markdown header per problem, in order.
- Dependencies only through `uv add` / `uv add --dev` (never pip, never hand-edited
  versions). Add only what the current problem needs and report what was added and why.
- Windows: use `pathlib` everywhere. Data dir: `DATA_DIR = Path(__file__).resolve().parents[1] / "data"`.
- Never `git commit` unless the user asks.

## Inline or src/? (be wise, no overkill)

Put it in `src/` when it is real logic: an algorithm, anything longer than ~10 lines,
anything with loops/branches that need thought, anything reused by a later problem,
any plot longer than a few lines, any table-building or verification routine.

Keep it inline in the notebook when it is a few plain lines: loading via one call,
a quick `print` of shapes, a one-line formula, a `pd.concat` of two tables, a loop
printing three thresholds. Wrapping a one-liner in a function is overkill.

## src/ code style (match list01)

- `from __future__ import annotations`; type hints on every public function.
- NumPy-style docstrings (`Parameters` / `Returns` / `Raises`) on public functions;
  explain *why* for non-obvious choices (e.g. why features are not standardised).
- Module-level constants in UPPER_CASE (`TOLERANCE = 1e-6`, `METHOD_ORDER = (...)`).
- Pure functions: inputs in, results out. No printing, no `plt.show()`, no hidden global state.
- Results meant for display are returned as `pandas.DataFrame` (one row per item,
  readable column names) so the notebook cell just ends with the call.
- Plot functions live in `plotting.py`, return the `Figure`, accept `ax: Axes | None = None`
  when a panel may be reused in a bigger figure, label axes, set titles, use one
  consistent palette, put a single shared legend on multi-panel figures.
- From-scratch implementations mirror the reference library interface
  (`fit`/`transform`/attributes with trailing underscore) and are checked against it.
- Randomness: `random_state: int = 0` parameters; the notebook passes the seed explicitly.
- Reuse earlier modules (`src.data`, earlier algorithms, `plot_2d_embedding(..., ax=ax)`)
  instead of duplicating code. Do not change earlier problems' code or cells unless asked.

## Notebook

Exact patterns with examples: `references/notebook-patterns.md`. In short:

- Top: `# Problem Set N: <title from the PDF>` and one line on the topic.
- Each problem starts with its own markdown header cell `## Problem K: <short title>`.
  All its content goes between that header and the next one.
- **[Theory]**: markdown only. State what is proven, then numbered **Step** blocks with
  LaTeX, a short *Intuition*, `$\blacksquare$`, and a small summary table when it helps.
- **[Python]**: (1) markdown **Approach and assumptions** bullets (data, preprocessing,
  conventions, seeds, deviations from the theory notation); (2) code cells, each with
  exactly one clear output; (3) markdown **Interpretation** / **Observations** quoting
  the actual numbers from the outputs.
- **[Analysis]**: markdown describing the metrics/evidence, code cells producing the
  evidence (in `src/`), then `### Discussion`: a **Short answer** first, bold-titled
  paragraphs, concrete numbers from the outputs, and the explicit link to the earlier
  problems the statement mentions. If the user says they write the discussion
  themselves, leave only `### Discussion\n\n*(to be written)*`.
- Code cells: imports at the top of the first code cell of each problem (each problem
  reads on its own); then calls and displays. Show tables as the last expression,
  figures as `fig = plot_xxx(...)` + `plt.show()`, verification as
  `assert table["passes"].all(), table[~table["passes"]]`.
- Later problems may reuse variables from earlier ones when the notebook runs top to
  bottom; say so in a short comment (`# X, y, results come from Problem 7`).
- Link [Python] problems to [Theory] ones: numerically confirm the derived results
  (e.g. a `theory_checks(...)` table) and compare from-scratch code against the library
  with the tolerance from the statement.
- Writing: plain, short sentences, "we", British spelling as in list01 (neighbour,
  normalise), no filler, no emojis, no restating the code. Every number in the text must
  come from the executed output. Read the outputs before writing.
- Edit the notebook with `nbformat` (or NotebookEdit), never by hand-writing JSON.
  Editing a markdown cell must not wipe other cells' outputs.

## Workflow per request

1. Read the problem(s) in the PDF, the relevant lecture slides and the existing
   `listNN/` (notebook, `src/`, prompts). Read earlier lists only for conventions.
2. If the request is not already a detailed spec: reply with a short plan
   (problem → approach → which `src/` module / inline → new deps) and wait for "go",
   unless the user said to go ahead.
3. Implement in order: dependencies → `src/` logic → `src/plotting.py` → notebook cells.
4. Verify (Definition of done), fix, rerun.
5. Report briefly: dependencies added, the key tables/numbers, any deviation from the
   spec and why, and what the user should double-check.

## Definition of done

From inside `listNN/`:

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace listN.ipynb
```

- The notebook runs end to end with no errors and is saved **with outputs**.
- Cells of other problems are unchanged (compare sources before/after).
- Random results are reproducible: run the key function twice with the same seed and
  confirm identical (or `allclose`) results. Known caveat: sklearn's Barnes-Hut t-SNE
  (and other OpenMP code) is deterministic per machine but depends on the CPU thread
  count, so quote such numbers to 2 decimals and mention it in the README.
- Every claim in the markdown matches the outputs.
- Warnings that only add noise (e.g. tqdm `IProgress`) are filtered in `src/__init__.py`,
  not with blanket `warnings.filterwarnings("ignore")` in the notebook.

## Submission (when the user asks to prepare a list for the teacher)

1. Write `listNN/README.md` from `references/readme-template.md` (problem → module table,
   setup with uv and a pip fallback, runtime, notes on network / data).
2. `uv export --format requirements-txt --no-hashes --no-emit-project > requirements.txt`
   (keeps the pip fallback pinned to `uv.lock`).
3. Test like the teacher would: zip the folder without `.venv`, `__pycache__`,
   `.ipynb_checkpoints`; unzip into a scratch dir; follow the README literally
   (`uv sync`, execute the notebook into a copy, check no errors); compare the key
   outputs with the saved ones. Do not re-execute the original notebook in place just
   for this, since runtimes in the text would change.
4. Fix only obvious typos in the user's prose and list them in the report.
