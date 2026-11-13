# Task: implement Problem 6 of Problem Set 3 (repo: analysis-of-unstructured-data)

Build prompt 5 of 5 (after `build4_problem5`).

## Context
- Problems 1–5 are solved. Don't change their `src` code or notebook cells. Put all Problem 6 content directly under the `## Problem 6 …` header (the last section).
- Same rules as before: logic in `list03/src/`, the notebook only imports and displays.
- Reuse what exists: the `embeddings` dict from Problem 5 (`{"word2vec (20NG)": w2v.wv, "GloVe (6B)": glove}`) is in scope when the notebook runs top to bottom, so don't retrain or reload. Reuse the plotting style from `src/plotting.py` (`SERIES_COLORS`, `INK*`, `_style_axes`).
- The assignment asks for **one paragraph** of discussion. Keep the probes few and readable; no extra diagnostic tables.

## Problem statement
Probe the embeddings for social bias (e.g. gendered analogies or association tests). Report what you find in one paragraph and comment on the risk of propagating such bias downstream.

## Step 1 - Dependencies
- None expected (numpy, pandas, gensim, matplotlib are installed). Verify, and report if you add anything.

## Step 2 - `src/bias.py` (new)
1. `WEAT_TESTS`: dict `name -> {"X", "Y", "A", "B"}` (space-separated, lower-cased word lists from Caliskan, Bryson & Narayanan, 2017, *Science* 356; cite it in a comment). Three tests, in this order:
   - `"WEAT 1: flowers vs. insects / pleasant vs. unpleasant (control)"`. Comment: a non-social control that a reliable embedding should pass.
     X `aster clover hyacinth marigold poppy azalea crocus iris orchid rose bluebell daffodil lilac pansy tulip buttercup daisy lily peony violet carnation gladiola magnolia petunia zinnia`,
     Y `ant caterpillar flea locust spider bedbug centipede fly maggot tarantula bee cockroach gnat mosquito termite beetle cricket hornet moth wasp blackfly dragonfly horsefly roach weevil`,
     A `caress freedom health love peace cheer friend heaven loyal pleasure diamond gentle honest lucky rainbow diploma gift honor miracle sunrise family happy laughter paradise vacation`,
     B `abuse crash filth murder sickness accident death grief poison stink assault disaster hatred pollute tragedy divorce jail poverty ugly cancer kill rotten vomit agony prison`.
   - `"WEAT 6: male vs. female names / career vs. family"`: X `john paul mike kevin steve greg jeff bill`, Y `amy joan lisa sarah diana kate ann donna`, A `executive management professional corporation salary office business career`, B `home parents children family cousins marriage wedding relatives`.
   - `"WEAT 7: math vs. arts / male vs. female terms"`: X `math algebra geometry calculus equations computation numbers addition`, Y `poetry art dance literature novel symphony drama sculpture`, A `male man boy brother he him his son`, B `female woman girl sister she her hers daughter`.
2. `weat(kv, X, Y, A, B, n_permutations=100_000, random_state=0) -> dict`: drop words missing from the vocabulary (return NaNs if any set has fewer than 2 words left). With unit vectors, `s(w, A, B) = mean_a cos(w, a) − mean_b cos(w, b)`; effect size `d = (mean_X s − mean_Y s) / std_{X∪Y} s` (`ddof=1`; `d > 0` means X goes with A and Y with B, |d| ≥ 0.8 is large). One-sided p-value over equal-size re-partitions of X ∪ Y (comparing `sum_X s` is enough, since `sum_X s − sum_Y s = 2 sum_X s − sum s`): exact enumeration with `itertools.combinations` when `comb(n, n_x) <= MAX_EXACT_PARTITIONS = 200_000`, otherwise `n_permutations` seeded random re-partitions with `p = (count + 1) / (n_permutations + 1)`. Return `effect_size`, `p_value`, `exact`, `n_words`, `n_missing`.
3. `weat_table(models, tests=None, random_state=0) -> DataFrame`: one row per test (index `test`, `None` = all tests), MultiIndex columns `(model, "d")` and `(model, "p")`.
4. `GENDER_PAIRS`: `(he, she), (his, her), (man, woman), (boy, girl), (father, mother), (son, daughter), (brother, sister), (male, female)`.
5. `gender_direction(kv, pairs=GENDER_PAIRS) -> np.ndarray`: the mean of the normalised differences `v(male) − v(female)` over the pairs in the vocabulary, scaled to unit length. Docstring: Bolukbasi et al. (2016) use the first principal component of the centred pairs; for pairs that agree well the mean difference points the same way and is simpler.
6. `OCCUPATIONS` (20 words, care/domestic to technical): `homemaker nurse receptionist librarian secretary hairdresser nanny teacher dancer doctor lawyer professor scientist programmer engineer architect surgeon mechanic carpenter pilot`.
7. `occupation_projection(models, occupations=OCCUPATIONS) -> DataFrame`: cosine of each occupation's unit vector with the model's gender direction; **positive = towards he/man, negative = towards she/woman**; NaN for missing words. One row per occupation (index `occupation`), one column per model.
8. `GENDER_ANALOGIES` (`a, b, c` triples): `man : doctor :: woman`, `man : engineer :: woman`, `man : scientist :: woman`, `man : boss :: woman`, `man : strong :: woman`, `woman : nurse :: man`.
9. `gendered_analogies(models, analogies=GENDER_ANALOGIES, allow_inputs=False) -> DataFrame`: each model's top answer. Standard protocol: `most_similar(positive=[b, c], negative=[a], topn=1)`, which never returns an input word. `allow_inputs=True`: `argmax` over all normalised vectors of `v̂(b) − v̂(a) + v̂(c)`, so inputs can win. Missing words → `"(<words> not in vocabulary)"`. Index `analogy` (`"a : b :: c : ?"`), one column per model. Docstring: for *man : doctor :: woman : ?* the unbiased answer is *doctor* itself, which the standard rule forbids; lifting the rule separates bias in the vectors from this artefact (Nissim, van Noord & van der Goot, 2020).

## Step 3 - `src/plotting.py` (add, keep the existing functions)
- `plot_gender_projection(projection) -> Figure`: one lollipop panel per model, side by side, sharing the occupations (y) and the horizontal scale (`±1.15 × max |value|`). Sort the occupations by the **last** model's projection (GloVe), most female-associated at the bottom. Each panel: vertical zero line, a horizontal line from 0 to each value plus a dot in that model's `SERIES_COLORS` slot, `"not in vocabulary"` in muted text for NaNs, the model name as title, `"← female-associated"` above the left end and `"male-associated →"` above the right end, x label `"cosine with the gender direction"`, `_style_axes(ax, grid_axis="x")`, occupation labels only on the first panel. Before labelling, check the orientation on a known pair (e.g. *he* must project positive) so the sides are not swapped.

## Step 4 - Notebook cells (under `## Problem 6 …`)
1. Markdown `**Probes.** We apply three probes to both embeddings from Problem 5.` and three short bullets in plain words, no formulas:
   - **WEAT** (Caliskan et al., 2017): whether two groups of target words (e.g. male and female names) differ in how close they are, on average, to two groups of attribute words (e.g. career and family); effect size $d$ as a standardised difference ($|d| \ge 0.8$ is large), p-value from a permutation test re-splitting the targets; two gender tests plus the non-social flowers/insects control that a reliable embedding should pass.
   - **Gender direction**: the average difference of eight male–female pairs (he–she, man–woman, father–mother, ...); the cosine of an occupation with it shows whether it lies closer to the male (positive) or female (negative) words.
   - **Gendered analogies** such as *man : doctor :: woman : ?*, answered as in Problem 5; the standard protocol never returns an input word, so for two of them we also show the answer when the inputs are allowed.
2. Code: imports (`matplotlib.pyplot`, `pandas`, `src.bias`, `src.plotting`); a one-line comment that `embeddings` comes from Problem 5; `weat_table(embeddings, random_state=0)`.
3. Code: `fig = plot_gender_projection(occupation_projection(embeddings))`; `plt.show()`.
4. Markdown, about 60–90 words, from the actual figure: what a point means; two or three patterns (which occupations sit at each end in GloVe, which lie near zero); how clear the word2vec pattern is (and how many words it lacks) compared with GloVe; these are associations learned from text, not properties of the jobs or evidence of who does them.
5. Code: `gendered_analogies(embeddings, GENDER_ANALOGIES)`.
6. Code: a comment, then `caveat = (("man", "doctor", "woman"), ("woman", "nurse", "man"))` and `pd.concat({"inputs excluded (standard)": gendered_analogies(embeddings, caveat), "inputs allowed": gendered_analogies(embeddings, caveat, allow_inputs=True)}, axis=1)`.
7. Markdown, about 70–100 words, from the actual tables: which answers follow occupational stereotypes and which do not; what word2vec returns; what changes when inputs are allowed and what that means. Conclude that analogy outputs reflect both the embedding and the evaluation procedure, so single examples are not proof of bias.
8. Markdown `### Discussion` followed by **one paragraph**, about 130–180 words, connecting the probes (not listing them): the main associations GloVe shows across WEAT, the gender direction and the analogies; one limitation of the probes (short hand-picked word lists, analogy results depend on the protocol); the small-corpus word2vec (points the same way on WEAT but fails the control and orders occupations erratically, so too noisy to trust, though not necessarily pure noise); one concrete downstream harm (e.g. a CV-screening model scoring applicants for engineering roles lower when their CVs contain female-associated words); why bias should be measured in the final application and mitigated. No subsections, no list of every effect size, no literature review.

## Step 5 - Verify
- Run the notebook from inside `list03/`. It must run end-to-end with outputs saved; Problem 1–5 cells unchanged.
- Check that every claim in the markdown matches the outputs (signs of `d`, which side each occupation is on, the analogy answers).
- Report the WEAT table, the occupations at both ends for each model, the analogy tables, the total runtime of the Problem 6 cells, and any deviation from this plan.
- For reference, the original solution got: GloVe d = 1.17 (control), 1.72 (WEAT 6), 1.28 (WEAT 7), all p < 0.01; word2vec d = 0.52 (control, p ≈ 0.18, fails), 0.97 (p ≈ 0.03), 1.50 (p ≈ 0.002). GloVe answers *nurse* to *man : doctor :: woman*, and *doctor* when inputs are allowed.
- Do not git commit.
