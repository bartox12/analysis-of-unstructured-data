# Refactor and simplify Assignment 3 notebook — Problems 4–6

## Context and objective

I have a Jupyter notebook, `list3.ipynb`, containing solutions to Assignment List 3 from the *Analysis of Unstructured Data* course. The assignment instructions are in `AUD_Lab_assignments.pdf`.

I want you to **directly edit and substantially simplify the notebook**, focusing exclusively on **Problem Set 3, Problems 4, 5, and 6** (Python and Analysis).

The solutions currently contain too much unnecessary code, numerical verification, secondary experiments, lengthy methodological explanations, and overly detailed conclusions.

The notebook is technically thorough, but I want it to be more concise, readable, and focused on the actual assignment requirements.

**The main principle is: every code cell, table, plot, and paragraph should have a clear purpose. If something neither satisfies an assignment requirement nor provides a meaningful, easily interpretable insight, remove it.**

This is not merely a formatting task. I want you to critically simplify the content, code, analyses, and narrative while preserving correctness.

## 1. First, inspect the project and assignment requirements

Before editing:

1. Read `AUD_Lab_assignments.pdf`, specifically **Problem Set 3, Problems 4–6**.
2. Read the entire `list3.ipynb`, including its code, Markdown, and saved outputs.
3. Inspect the relevant `src` modules used by the notebook.
4. Identify dependencies between notebook cells so removing one experiment does not break another.
5. Preserve all essential assignment requirements.

For reference, the mandatory requirements are:

**Problem 4 — TF-IDF classification**
- Preprocess text with NLTK or spaCy.
- Build a TF-IDF pipeline using `TfidfVectorizer`.
- Train Multinomial Naïve Bayes and logistic regression.
- Report precision, recall, and F1.
- Identify the most informative features per class.

**Problem 5 — word2vec and GloVe**
- Train word2vec with gensim on the same corpus.
- Load pretrained GloVe separately.
- Return nearest neighbours of five query words for both models.
- Evaluate both models on an analogy dataset.

**Problem 6 — social bias**
- Probe embeddings for social bias using appropriate tests.
- Report findings in one paragraph.
- Discuss the risk of propagating such bias downstream.

Everything beyond these requirements should justify its presence by providing an important, clearly interpretable insight.

## 2. General editing rules

Follow these throughout the notebook.

### A. Remove unnecessary validation and diagnostic experiments

Remove demonstrations whose only purpose is to show that our implementation agrees numerically with another implementation or a theoretical formula, unless the assignment explicitly requires such verification.

Examples:
- Reconstructing TF-IDF numerically and comparing it against scikit-learn.
- Checking a custom Naïve Bayes implementation against scikit-learn.
- Comparing our analogy implementation with gensim on 200 questions.
- SGNS finite-difference gradient checks.
- Extra correlations and numerical consistency checks that do not provide meaningful interpretation.

Keep internal correctness safeguards when they are essential for reliable execution, but do not present them as separate experiments in the notebook.

### B. Reduce redundant results

If two tables or plots communicate essentially the same finding, retain the clearer one.

Prefer:
- One clear table instead of several overlapping tables.
- A compact visualization instead of a large multi-panel figure when it communicates the result better.
- A few representative examples instead of exhaustive outputs.
- Summary metrics instead of dozens of secondary statistics.

Avoid displaying numerical information that is not subsequently interpreted.

### C. Make Markdown explanations substantially shorter

This is a major part of the task.

The current notebook contains long methodological descriptions and conclusions that often read more like a research paper than a university assignment solution.

I want the revised notebook to be **concise but still academically rigorous**.

In particular:

- Shorten introductory "Approach and assumptions" sections.
- Describe the important methodological choices, not every implementation detail.
- Avoid repeating numerical results already visible in tables or figures.
- Avoid deriving mathematical relationships when the assignment does not ask for the derivation.
- Prefer clear explanations in ordinary English to unnecessary formulas.
- Use mathematical notation only when it meaningfully improves clarity.
- Avoid long lists of individual numerical values.
- Avoid explaining obvious code operations.
- Remove repeated references to earlier theoretical problems when they are not relevant to the current task.

The objective is **not** to remove mathematical rigor where necessary, but to avoid unnecessary mathematical formalism in empirical observations.

For example, instead of describing several IDF formulas and floating-point errors, explain in a sentence how TF-IDF gives more importance to document-specific terms.

Instead of listing many WEAT statistics in prose, explain what the results suggest about gender associations.

### D. Improve observations and conclusions

For every observation section, use the following logic:

1. What did we observe?
2. What does it tell us about the method or dataset?
3. Is there an important limitation or caveat?

Do not mechanically create three numbered points. Write natural, coherent paragraphs.

Each paragraph should communicate a distinct finding.

Prefer specific, descriptive conclusions over broad generic statements or long mathematical explanations.

For example:

**Too technical:**
"The reconstructed TF-IDF matrix matches the theoretical formula to a maximum absolute error of approximately 3 × 10⁻¹⁶."

**Better:**
"Raw word counts are dominated by common words, whereas TF-IDF highlights terms that better distinguish the document's topic."

Another example:

**Too verbose:**
A paragraph listing individual WEAT effect sizes, p-values, projection scores, and analogy outputs, followed by another paragraph repeating the same findings.

**Better:**
"GloVe exhibits clear gender associations: career and technical terms tend to be closer to male-associated words, while family-related terms are closer to female-associated words. These patterns may reinforce stereotypes in downstream applications."

Retain a small number of important numerical results when they are genuinely useful.

Do not replace meaningful analysis with generic statements such as "Model A performed better than Model B" without explaining why that difference is interesting.

### E. Preserve scientific correctness

- Do not invent results or numerical values.
- Do not change the mathematical meaning of existing analyses.
- Do not introduce unsupported causal explanations.
- Distinguish observations from plausible hypotheses.
- Avoid drawing strong conclusions from weak or noisy evidence.
- Maintain proper train/test separation.
- Preserve reproducibility and random seeds where relevant.
- Keep the notebook in English.

## 3. Specific changes — Problem 4

Original notebook section: approximately cells 4–18 (zero-based indices).

### 3.1. Simplify the introductory explanation

The current "Approach and assumptions" is overly detailed.

Keep a brief explanation of:
- The dataset and six selected categories.
- NLTK preprocessing.
- TF-IDF representation.
- The two classifiers and the training/evaluation strategy.
- How informative features are identified, if necessary.

Avoid lengthy explanations of smoothing formulas, normalization, centered weights, and implementation choices.

A few concise bullet points or short paragraphs are enough.

### 3.2. Remove TF-IDF numerical verification

Original cells 8–9:

- Remove the Markdown titled "Check of Problem 1".
- Remove the `tfidf_checks` experiment.
- Remove its numerical comparison table.
- Remove the corresponding explanatory text in Observations.
- Remove unused imports.

These checks are not required for Problem 4.

### 3.3. Remove or considerably simplify the IDF table

Original cell 10:

The `idf_table` illustrates the difference between frequent and more topic-specific words.

It is conceptually interesting, but largely overlaps with the following raw-counts-versus-TF-IDF experiment.

**Preferred approach: remove the IDF table**, provided the next example adequately demonstrates the benefit of TF-IDF.

Avoid presenting the same basic insight twice.

### 3.4. Keep raw counts vs. TF-IDF, but shorten it

Original cells 11–12:

I specifically want to retain this comparison because it is intuitive and easy to interpret.

However:
- Reduce the displayed number of terms if appropriate (e.g. 6–8 rather than 10).
- Shorten the introductory Markdown.
- Avoid unnecessary document metadata or technical details.
- Add only a brief observation explaining the difference.

The important message is that raw frequencies favor common words, while TF-IDF gives greater weight to more distinguishing terms.

### 3.5. Remove custom Naïve Bayes verification

Original cells 13–14:

Remove:
- The "Check of Problem 2" explanation.
- `compare_with_sklearn`.
- Comparisons of class priors, log probabilities, decision scores, prediction probabilities, and predicted classes.
- All accompanying numerical outputs.

This is unnecessary for an assignment asking us to train and evaluate classifiers.

### 3.6. Simplify model selection reporting

Original cell 7:

Retain meaningful information about hyperparameter selection and cross-validation.

However, simplify `cv_summary` if necessary.

A compact summary of:
- Selected hyperparameters.
- Cross-validation macro-F1.

is sufficient.

Remove search times and other secondary columns unless they serve a clear purpose.

Do not unnecessarily change the training strategy or hyperparameter search itself.

### 3.7. Preserve the classification metrics

Original cell 15:

Keep the required precision, recall, and F1 results, preferably class-wise and with an overall or macro summary.

Accuracy can be included as an additional useful metric.

Ensure both models can be easily compared.

### 3.8. Keep useful confusion-matrix information

Original cell 16:

The confusion matrices provide meaningful additional information, especially for identifying which classes are confused.

Retain them if they remain readable and useful.

Simplify the visualization if necessary, and avoid excessive commentary.

One short observation identifying the most significant confusion pattern is enough.

### 3.9. Simplify informative features

Original cell 17:

Keep the most informative terms for each class and each model, as explicitly required.

Remove:
- `weight_agreement`.
- Spearman correlations between feature-weight vectors.
- Shared-top-10 overlap statistics.

Consider showing 6–8 informative features per class rather than 10, provided the essential comparison remains clear.

### 3.10. Rewrite the Observations section

Original cell 18 is much too long.

Remove:
- Numerical confirmations of Problems 1 and 2.
- Detailed calculations explaining the selected smoothing parameter.
- Repeated IDF values.
- Spearman correlations and overlap statistics.
- Lengthy discussions that repeat the tables.

Keep the most informative findings:
1. Which classifier performed better and by approximately how much.
2. Which classes were easiest/hardest and why some classes were confused.
3. How the most informative features differ between the models.
4. The interesting observation that residual author signatures can become predictive features, causing the model to learn who wrote the text rather than solely what it is about.

The author-signature observation is particularly valuable, but the current explanation is too long. Explain it in two or three clear sentences, without reproducing the entire signature or excessive statistics.

**Target: approximately 100–150 words for the entire Problem 4 Observations section**, provided all essential insights remain.

## 4. Specific changes — Problem 5

Original notebook section: approximately cells 19–34.

### 4.1. Shorten the methodology

Original cell 20:

Keep the essential information:
- Corpus.
- Tokenization.
- word2vec training configuration.
- Pretrained GloVe.
- Five query words.
- Analogy evaluation protocol and fair comparison.

Remove excessive details about the negative sampling distribution, candidate limits, exact internal validation methods, and other implementation details.

Do not include a lengthy explanation of SGNS theory here.

Remove the "Link to Problem 3" subsection describing gradient checks and shifted PMI.

Those experiments will be removed.

### 4.2. Simplify corpus and vocabulary diagnostics

Original cells 21–22:

Keep only useful dataset-size information if it helps interpret the results.

The current vocabulary summary can be reduced or removed if it adds little beyond confirming that the models loaded correctly.

Training time is optional and probably unnecessary.

Retain enough information to understand why the locally trained word2vec model has much less training data than pretrained GloVe.

### 4.3. Preserve nearest neighbours

Original cell 23:

This is explicitly required.

Keep five query words for each embedding model.

Consider reducing from eight neighbours per query to five, making the results more readable.

Ensure the comparison remains easily interpretable.

### 4.4. Shorten the manually selected analogies

Original cell 24:

This is an interesting additional experiment, so I would like to keep a compact version.

Reduce six examples to approximately three or four, choosing examples that demonstrate different behaviors.

Prefer:
- A successful semantic analogy.
- A grammatical analogy.
- An example illustrating a failure or ambiguity, such as `microsoft : windows :: apple : ?`.

Use compact output, preferably top-1 predictions rather than top-3 lists where possible.

Add a brief interpretation of the most interesting result.

Avoid duplicating information from the main analogy accuracy evaluation.

### 4.5. Remove the comparison with gensim

Original cell 25:

Remove the `compare_with_gensim` numerical verification entirely.

However, **preserve any code needed to load the analogy questions and determine common coverage**.

Removing the check must not break the subsequent evaluation.

### 4.6. Simplify analogy evaluation

Original cells 26–28:

Currently, analogy evaluation is presented using:
- Coverage-based model-specific accuracy.
- Accuracy on questions both models can answer.
- Fourteen category-level results.
- A detailed plot.

This is excessive.

Keep a compact, fair comparison showing:
- Coverage of the analogy test set.
- Accuracy on the questions both models can answer.
- Optionally, semantic versus syntactic accuracy.

A small, readable table or a simple chart is preferable.

Do not show fourteen category-level rows unless there is an especially important reason.

If a chart merely repeats the summary table, remove the chart.

The distinction between **coverage** and **accuracy on shared questions** is important and should remain clear.

### 4.7. Remove SGNS gradient checks

Original cells 29–30:

Remove:
- The gradient-check introduction.
- `gradient_check`.
- The numerical gradient errors.
- References to these checks in subsequent observations.

This verification belongs to the theoretical discussion of Problem 3, not to Problem 5.

### 4.8. Remove shifted PMI experiments

Original cells 31–33:

Remove the entire shifted-PMI analysis:
- Its lengthy theoretical Markdown.
- `shifted_pmi_check`.
- PMI comparison tables.
- The PMI-versus-dot-product plot.
- Related explanations in the Observations section.

Although mathematically interesting, this is an extensive experiment unrelated to the central deliverables of Problem 5.

Do not move it elsewhere in the main notebook. We want a shorter solution, not an appendix containing all removed experiments.

Do not delete reusable functions from the project solely because they are no longer used in this notebook.

### 4.9. Rewrite the Observations section

Original cell 34:

Keep the following insights:

1. **Nearest neighbours:** the differences between the corpus-specific word2vec model and the general-purpose GloVe model, particularly examples of polysemy or domain-specific vocabulary.
2. **Analogies:** the substantial performance gap between the two models, including one or two representative numbers.
3. **Interpretation:** corpus size and domain differences are plausible reasons for that gap, but the experiment does not isolate the effects of training method, corpus size, and corpus domain.
4. **Limitation:** locally trained word2vec has weaker vocabulary coverage and may produce unreliable results for infrequent words.

Remove:
- Verification against gensim.
- All shifted PMI results.
- All gradient-check results.
- Long lists of individual analogy-category scores.
- Repetitions of table contents.
- Overly strong causal claims.

**Target: approximately 100–150 words, using two or three short paragraphs.**

The writing should be predominantly descriptive rather than mathematical.

## 5. Specific changes — Problem 6

Original notebook section: approximately cells 35–40.

The assignment explicitly requires a short discussion in one paragraph. The current solution is far more extensive than necessary.

I want a meaningful but much smaller analysis.

### 5.1. Simplify the introduction to bias probes

Original cell 36:

Remove lengthy mathematical definitions, full WEAT formulas, permutation-enumeration details, sample counts, and extensive explanations of the testing procedures.

Use short, accessible explanations.

For example:
"WEAT measures whether different groups of words have systematically different associations with attributes such as career and family."

Explain only the information required to interpret the results.

### 5.2. Reduce the number of WEAT tests

Original cell 37:

Currently, we evaluate:
- WEAT 1, flowers/insects (non-social control).
- WEAT 6, gender/career–family.
- WEAT 7, gender/math–arts.
- WEAT 8, gender/science–arts.

We do not necessarily need all four.

Prefer:
- One relevant gender-association test.
- Optionally one additional gender test with a distinct interpretation.
- The non-social control if it helps assess whether the smaller word2vec embedding is reliable.

Present the results in a compact table with the key effect sizes and p-values.

Avoid excessive numerical detail.

If changing the helper function is required to select a subset of tests, make the smallest reasonable change.

### 5.3. Shorten occupation projections or remove them

Original cell 38:

The gender-direction experiment is interesting, especially because it offers a simple interpretation of how occupations are represented.

However, the current analysis is too extensive.

If retaining it:
- Show a smaller, carefully selected group of occupations.
- Use a clean and compact figure.
- Avoid unnecessary baseline calculations unless they materially improve interpretation.
- Remove the Spearman correlation comparing occupation projections between models.
- Do not print additional statistics that repeat the graph.

If the projection plot adds little beyond the WEAT and analogy results, remove it entirely.

Prioritize clarity rather than retaining every possible bias probe.

### 5.4. Retain a small number of gendered analogies

Original cell 39:

This is another interpretable experiment.

Keep approximately two or three representative examples rather than seven.

In particular, preserve the important methodological caveat that standard analogy evaluation excludes input words from the candidate answers.

The comparison between standard analogies and allowing the input occupation as an answer is meaningful because it demonstrates how apparently stereotyped outputs can be affected by the evaluation procedure.

Show this with one or two compact examples, not a large table.

Avoid unnecessarily duplicating the message already communicated by WEAT or occupation projections.

### 5.5. Rewrite the Discussion as ONE paragraph

Original cell 40:

Replace the entire multi-section discussion with **one coherent paragraph**, approximately 120–170 words.

This is explicitly requested by the assignment.

The paragraph should:
- Briefly describe the strongest observed bias patterns.
- Distinguish the clearer GloVe findings from the less reliable small-corpus word2vec findings.
- Explain one important caveat of the probes.
- Give one concrete example of possible downstream harm.
- Briefly explain why such associations should be monitored or mitigated.

Avoid:
- Repeating every effect size and p-value.
- Listing all occupation projections.
- Reproducing analogy tables in prose.
- Explaining the mathematical connection to shifted PMI.
- Discussing multiple downstream applications.
- Lengthy literature reviews of debiasing methods.
- Multiple subsections such as "Short answer", "What we find", "Risks", "Caveats", and "Relation to previous problems".

One well-written paragraph should communicate the findings and their significance.

Also, be careful with interpretation: failure of the non-social WEAT control makes results from the small word2vec model less reliable, but does not by itself prove that all its observed associations are noise.

## 6. Preserve important, readable extensions

I do not want you to remove everything that is technically optional.

Some additions genuinely improve the educational value of the notebook.

Prioritize keeping:

**Problem 4**
- Raw counts vs. TF-IDF.
- Useful confusion-matrix information.
- Informative feature comparisons.
- The short observation about author signatures and potential dataset artifacts.

**Problem 5**
- Domain-specific nearest-neighbour differences.
- A few representative analogies.
- A fair coverage-aware analogy comparison.

**Problem 6**
- One or two informative gender-bias probes.
- One concrete example showing a meaningful limitation of analogy-based bias evaluation.

Everything else should be removed, shortened, or consolidated unless there is a strong reason to keep it.

## 7. Code quality and notebook organization

After removing experiments:

- Remove unused imports.
- Remove variables that are no longer referenced.
- Remove dead code from notebook cells.
- Fix references to deleted variables.
- Consolidate trivial cells where this improves readability.
- Keep logical section boundaries.
- Avoid excessive print statements.
- Avoid debugging-style output.
- Keep tables and figures compact and readable.
- Keep only figures that contribute something beyond the surrounding tables.
- Preserve reproducible training and evaluation.
- Use existing project helpers when appropriate.
- Do not introduce large new dependencies.
- Do not perform unnecessary rewrites of the `src` package.

The notebook should read naturally from top to bottom:

**Problem 4:** Data and preprocessing → TF-IDF and classifiers → results → informative features → concise observations.

**Problem 5:** Embedding models → nearest neighbours → analogy evaluation → concise observations.

**Problem 6:** Bias probes → results → one-paragraph discussion.

There should be a logical progression, without unrelated experiments interrupting the main solution.

## 8. Important scope and methodological constraints

### Keep Problems 1–3 unchanged

The first three problems are theoretical solutions.

**Do not shorten, rewrite, or remove their derivations.**

This task concerns only Problems 4–6.

You may remove unnecessary references to Problems 1–3 from the later empirical sections.

### Do not silently change experiments

Avoid changing:
- Dataset splits.
- Model architectures.
- Training hyperparameters.
- Preprocessing rules.
- Evaluation definitions.

unless absolutely necessary for functionality or to meet the assignment requirements.

There is one existing issue to flag:

Problem 4 uses six 20 Newsgroups categories, while Problem 5 trains word2vec on all twenty categories, including train and test posts. The assignment says "same corpus."

Do not silently change the datasets as part of this cleanup, as that would affect the trained embeddings and all results. Instead, clearly identify this mismatch in your final summary and recommend the smallest correction needed for strict compliance. Keep the main task focused on simplification.

### Preserve output integrity

If you change code that produces a table or figure, ensure its saved notebook output corresponds to the updated code.

Never leave outdated outputs attached to modified cells.

Do not fabricate or manually invent numerical results.

If an expensive experiment cannot be rerun, explain this clearly and leave its output in an appropriate state.

## 9. Validation after editing

After completing the edits:

1. Verify that the notebook is still valid `.ipynb` JSON.
2. Confirm that all imports and references remain consistent.
3. Check that deleted cells are not referenced by later cells.
4. Confirm that Problems 4–6 still satisfy every requirement from the PDF.
5. Run the notebook, or the affected sections, if reasonably possible in the available environment.
6. If execution requires unavailable downloads or dependencies, perform static checks and clearly state what remains unverified.
7. Check that all retained tables, plots, and explanations are consistent.
8. Confirm that Problems 1–3 have not been changed.

Make the edits directly in `list3.ipynb`. Do not create a parallel, completely rewritten notebook unless necessary.

## 10. Expected final deliverable

I want an edited notebook that is:

- **Substantially shorter**, both in code and text.
- Fully aligned with the assignment requirements.
- Easy to read and navigate.
- Academically appropriate.
- Clear in its interpretation of results.
- Free of unnecessary numerical checks.
- Free of repetitive tables and plots.
- Reproducible and technically correct.
- Still sufficiently detailed to demonstrate understanding of the methods.

Do not aim for the smallest possible notebook at the expense of interpretation. Aim for the **shortest notebook that provides a complete, convincing, and insightful solution**.

After editing, provide a concise summary containing:

1. Which major cells or experiments were removed.
2. Which optional experiments were kept and why.
3. How the Markdown explanations and conclusions were shortened.
4. Whether every requirement of Problems 4–6 is still satisfied.
5. Any execution issues or unresolved methodological concerns.

Also report the original and final number of notebook cells.

**Please implement these changes, rather than only providing recommendations or a proposed plan.**
