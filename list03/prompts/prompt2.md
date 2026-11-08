# Refine Assignment 3 notebook — Restore selected insightful experiments

I want to make a few improvements to the recently simplified `list3.ipynb`.

The previous refactoring successfully removed unnecessary numerical checks, redundant analyses, and lengthy explanations. However, I think we simplified some parts too aggressively and lost a few genuinely interesting, interpretable results.

I want to restore some of that analytical depth **without bringing back the verbosity of the original notebook**.

Please make the following changes directly to the current notebook.

## 1. Restore the gender direction plot with occupations (Problem 6)

The original notebook contained an interesting experiment that projected occupation embeddings onto a gender direction defined from male–female word pairs.

I want this visualization restored.

### Implementation

- Reuse the original approach and existing helper functions from `src`, if available.
- Define the gender direction using the original male–female word pairs.
- Project occupation embeddings onto this direction using cosine similarity or the original normalized projection method.
- Include approximately 15–20 occupations, preferably using the original occupation list.
- Compare the results for both locally trained word2vec and pretrained GloVe.

### Visualization

Create a clean, readable visualization with:

- Occupations on the vertical axis.
- Gender-direction projection scores on the horizontal axis.
- Negative and positive values indicating the two ends of the gender direction.
- A clear zero reference line.
- Labels indicating which side is male-associated and which is female-associated.
- A straightforward comparison between word2vec and GloVe.

Prefer one coherent figure, perhaps two aligned panels with the same horizontal scale, rather than multiple separate plots.

Sort occupations in a meaningful way, preferably according to GloVe projections.

Retain enough occupations to reveal meaningful patterns, including technical, medical, educational, and traditionally gender-associated professions.

Do not add unnecessary statistics or supplementary numerical tables. The figure should communicate the main finding by itself.

### Interpretation

Add a concise Markdown explanation of approximately 60–90 words that:

1. Explains what the projections represent.
2. Highlights two or three interesting patterns visible in the results.
3. Compares the stability or clarity of the patterns between word2vec and GloVe.
4. States that these are associations learned from text, not inherent characteristics of occupations or evidence of actual gender distributions.

Avoid reproducing individual projection values in the text unless particularly informative.

Important: verify the orientation of the gender direction before labeling positive and negative scores. Do not accidentally reverse the meaning of the axis.

## 2. Expand the analogy examples in Problem 5

The simplified notebook may contain too few examples of semantic and syntactic analogies.

I would like to show approximately **6–8 carefully chosen analogies**.

Choose examples that illustrate different properties of embedding models, such as:

- Country–capital relations.
- Gender-related word relations.
- Comparative or superlative adjective forms.
- Verb tense transformations.
- Domain-specific relationships.
- Polysemy or ambiguous words.
- Cases where GloVe performs well but word2vec fails.

Prefer reusing the examples from the original notebook, adding new ones only when useful.

### Presentation

Use a compact comparison table with columns such as:

| Analogy | word2vec prediction | GloVe prediction |
|---|---|---|

Include the expected answer where it helps interpretation.

Show the most relevant prediction for each model rather than printing long lists of nearest candidates.

If a word is outside the vocabulary, report that explicitly rather than treating it as an incorrect prediction.

Choose examples based on the actual model outputs, not assumptions about which model should succeed.

### Interpretation

After the table, add a short paragraph of approximately 70–100 words.

Focus on:
- Which types of relations the models capture.
- Interesting differences in predictions.
- The effect of corpus size or domain as a plausible explanation.
- At least one noteworthy failure or ambiguity.

Avoid explaining the mathematical 3CosAdd formula again. The aim is to interpret model behavior, not derive the analogy method.

Keep the main quantitative analogy benchmark and coverage-aware comparison unchanged.

## 3. Expand the gendered analogies in Problem 6

I would also like more examples in the social-bias analysis.

Include approximately **4–6 gender-related analogies** rather than just two or three.

Choose a diverse selection:
- Occupation-related analogies.
- Traditional gender associations.
- An example of a potentially stereotypical prediction.
- An example where the expected stereotype does not appear.
- An example showing that analogy predictions can depend on candidate exclusion.

Use both word2vec and GloVe where vocabulary coverage permits.

### Preserve the methodological caveat

One especially interesting observation from the original notebook was that standard analogy evaluation excludes the input words from the candidate set.

For example, a stereotypical output such as `nurse` may appear partly because `doctor` was excluded from the candidate answers.

I want to preserve this insight.

Show a small comparison for one or two carefully chosen analogies:
- Standard analogy evaluation (input words excluded).
- Evaluation allowing the input words as candidates.

Do not turn this into another large diagnostic experiment.

### Interpretation

Include a short explanation of what the examples illustrate, emphasizing that analogy outputs are influenced by both the learned embeddings and the evaluation procedure.

Avoid presenting isolated analogy outputs as definitive proof of social bias.

## 4. Maintain a concise Problem 6 discussion

The assignment asks for a one-paragraph discussion of social bias and downstream risks.

Keep the final Discussion as **one coherent paragraph of approximately 130–180 words**.

Now that the occupation projection plot and additional analogies are included, incorporate their most important findings into this paragraph.

Do not repeat every result.

The final discussion should cover:
- The main bias associations observed.
- The occupational gender-direction patterns.
- What the gendered analogies reveal.
- An important limitation of the methodology.
- One concrete example of downstream harm.

Avoid duplicating the separate brief explanations beside the figures and tables.

The paragraph should connect the results into an overall conclusion rather than list them independently.

## 5. Preserve the improvements from the previous refactoring

This is an incremental improvement, **not a request to revert the previous simplification**.

Do not restore:

- TF-IDF numerical checks.
- Custom Naïve Bayes verification.
- Gensim implementation comparisons.
- SGNS gradient checks.
- Shifted PMI experiments.
- Large category-by-category analogy result tables.
- Unnecessary Spearman correlations.
- Excessive mathematical derivations.
- Verbose methodological descriptions.

Do not substantially change Problem 4 or the theoretical Problems 1–3.

Keep the notebook readable, concise, and focused on the assignment.

The goal is to bring back a few insightful experiments that make the results more interesting, rather than to increase the amount of code for its own sake.

## 6. Technical requirements

- Inspect the current notebook and relevant existing `src` modules before making changes.
- Prefer reusing existing functionality to introducing new implementations.
- Keep the added code reasonably short and modular.
- Preserve the existing training configurations, embeddings, and datasets.
- Do not retrain models unnecessarily.
- Maintain consistent tables, plots, and Markdown styling.
- Keep all text in English.
- Avoid hardcoded numerical results that may disagree with actual execution.
- Make sure the notebook runs correctly after the changes.
- Update saved outputs to reflect modified code whenever execution is possible.

If the old occupation-direction implementation is available in the project, reuse it rather than rebuilding it from scratch.

## 7. Expected result

I want a notebook that strikes a better balance between **conciseness and analytical depth**.

The result should have:

1. A restored, visually informative gender-direction plot with approximately 15–20 occupations.
2. More varied semantic and syntactic analogy examples in Problem 5.
3. More representative gendered analogy examples in Problem 6.
4. A compact demonstration of how excluding input words affects analogy predictions.
5. Short, descriptive interpretations of these results.
6. A concise final discussion connecting the bias experiments.

**Please edit the notebook directly, validate the modified sections, and summarize what you changed.**

Do not make unrelated changes to the rest of the notebook.
