# Task: write the theory answers for Problems 1–3 of Problem Set 3 (repo: analysis-of-unstructured-data)

Build prompt 2 of 5 (after `build1_setup`).

## Context
- `list03/list3.ipynb` has a title cell and empty `## Problem 1 …` to `## Problem 6 …` header cells. Put each answer directly under its header, before the next header. Do not touch the other cells.
- These are [Theory] problems: Markdown only, no code. One markdown cell per problem (the header cell itself can hold the answer, as in `list01/list1.ipynb`).
- Read the problem text in `materials/AUD_Lab_assignments.pdf` (Problem Set 3, Problems 1–3). The PDF formulas extract badly, so reconstruct them carefully. Use the lecture notation from `materials/1_Introduction_and_the_Representation_Paradigm.pdf` (tf as the count normalised by document length).
- Style as in `list01`: start with **What we derive:** (one sentence), then numbered `### Step k: …` blocks, a plain-words sentence after each important formula, boxed key results (`\boxed{}`), `$\blacksquare$` at the end of each proof, short *Intuition:* lines, and a summary table where there are several results. Plain short sentences, British spelling, no filler.
- Problems 4–6 only train and evaluate models. Do not promise numerical checks of these derivations in later problems. A forward reference is fine only for something those problems actually use (e.g. "`TfidfVectorizer` in Problem 4 uses a smoothed variant").

## Problem statements
1. Derive the TF-IDF weight $w_{t,d} = \mathrm{tf}_{t,d}\cdot\log\frac{N}{\mathrm{df}_t}$. Give an information-theoretic reading of the $\log(N/\mathrm{df}_t)$ term and explain why raw term frequency alone is a poor weighting.
2. (Multinomial Naïve Bayes.) Derive the MAP parameter estimates with Laplace smoothing, $\hat\theta_{t\mid c} = \frac{n_{t,c} + \alpha}{\sum_{t'} n_{t',c} + \alpha|V|}$. Show that the resulting decision rule is linear in the word-count vector, and compare its form to logistic regression.
3. (Skip-gram with negative sampling.) Write the SGNS objective $\log\sigma(v_c^\top v_w) + \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n}\big[\log\sigma(-v_{w_i}^\top v_w)\big]$. Derive the gradient w.r.t. a centre-word vector $v_w$, and explain informally why SGNS approximates a shifted PMI factorisation.

## Problem 1: TF-IDF weighting
- **Notation (as in the lecture):** $N$ documents, vocabulary $V$, $n_{t,d}$ occurrences, $|d|$ the length, $\mathrm{tf}_{t,d} = n_{t,d}/|d|$, $\mathrm{df}_t$ the number of documents containing $t$.
- **Step 1: what a term weight has to do.** Large when frequent in $d$ and rare in the corpus; the two effects are independent, so $w_{t,d} = \mathrm{tf}_{t,d}\cdot g(\mathrm{df}_t)$ with $g$ decreasing.
- **Step 2: $\log(N/\mathrm{df}_t)$ is the information carried by "$t$ occurs".** For a uniformly random document, $P(t\in D) = \mathrm{df}_t/N$, so the self-information is $\mathrm{idf}_t$. Show the same number as an entropy reduction $H(D) - H(D\mid t\in D) = \log N - \log \mathrm{df}_t$. Bits reading with $\log_2$. Intuition: a term in every document gives 0, a term in one document gives $\log N$; logs make independent terms add. Small table: $\mathrm{df}_t \in \{1000, 100, 10, 1\}$ with $N = 1000$, in nats and bits.
- **Step 3: tf·idf is each pair's share of the mutual information between terms and documents** (after Aizawa, 2003). Draw one token uniformly from the corpus: $P(t,d) = n_{t,d}/M$, $P(d) = |d|/M$, $P(d\mid t) = n_{t,d}/n_t$. Write $I(T;D)$. Assumptions (A1) equal document lengths, (A2) a term occurs equally often in every document containing it. Under them $I(T;D) = \frac1N\sum_d\sum_t \mathrm{tf}_{t,d}\log\frac{N}{\mathrm{df}_t}$, so $w_{t,d}$ is the contribution of the pair $(t,d)$. Note that real text breaks the assumptions (lengths differ, words are bursty), which motivates length normalisation and sublinear tf.
- **Step 4: why raw term frequency alone is a poor weight** (equivalent to $\mathrm{idf}_t = 1$): (1) function words dominate (Zipf), (2) no discriminative power (e.g. "would" vs. "orbit"), (3) length bias, (4) burstiness, with $1 + \log n_{t,d}$ as the usual fix.
- **Summary table**: idf, tf·idf, raw tf with formula and reading.
- One closing line: scikit-learn's `TfidfVectorizer` (Problem 4) uses the smoothed $\mathrm{idf}_t = \ln\frac{1+N}{1+\mathrm{df}_t} + 1$, and what the smoothing means.

## Problem 2: Multinomial Naïve Bayes
- **Setup:** classes $c$, count vector $x$, length $L$; multinomial model $P(x\mid c) = \frac{L!}{\prod_t x_t!}\prod_t\theta_{t\mid c}^{x_t}$, prior $\pi_c$; $n_{t,c}$ and $N_c$. Name the "naïve" independence assumption.
- **Step 1: the likelihood** $\sum_c\sum_t n_{t,c}\log\theta_{t\mid c}$ + const; classes separate.
- **Step 2: Dirichlet prior and the MAP estimate.** Symmetric Dirichlet with parameter $\beta$, $\alpha = \beta - 1$, Lagrange multiplier, boxed $\hat\theta_{t\mid c}$, concavity argument. Bullets: Laplace is $\alpha = 1$ (Dirichlet $\beta = 2$, or the posterior mean under the uniform prior); Lidstone $0<\alpha<1$; $\alpha\to0$ gives the MLE; why smoothing is needed (an unseen word gives $\log 0 = -\infty$); $\hat\pi_c = N_c/N$.
- **Step 3: the decision rule is linear in $x$.** $\arg\max_c[w_c^\top x + b_c]$ with boxed $w_{c,t} = \log\hat\theta_{t\mid c}$, $b_c = \log\hat\pi_c$; hyperplane boundaries; posterior in softmax form.
- **Step 4: comparison with logistic regression.** Two-class log-odds give the sigmoid form; for $K$ classes both are $\operatorname{softmax}(Wx+b)$. Table: type (generative vs. discriminative), decision function, weights, fitting (closed form vs. iterative convex, gradient $p - y$ from Problem Set 2), regulariser (pseudo-count $\alpha$ vs. L2 $1/C$), correlated words, data needed (Ng & Jordan, 2002).
- *Intuition:* NB is a linear classifier whose weights are set by a rule of thumb instead of being optimised.
- *Assumption for Problem 4:* we feed NB TF-IDF weights instead of counts; formulas unchanged, rule stays linear, no longer a proper multinomial likelihood (common choice, Rennie et al., 2003).

## Problem 3: Skip-gram with negative sampling
- **Setup:** two vectors per word (centre $v_w$, context vectors; gensim stores them in `wv` and `syn1neg`). The objective $\ell(w,c)$ with $P_n(w)\propto\mathrm{count}(w)^{3/4}$. Reading: logistic regression telling real pairs from noise pairs.
- **Step 1: sigmoid facts** $\sigma' = \sigma(1-\sigma)$, $\frac{d}{dx}\log\sigma(x) = \sigma(-x)$, $\frac{d}{dx}\log\sigma(-x) = -\sigma(x)$.
- **Step 2: gradient w.r.t. $v_w$.** Positive term, negative terms (gradient through the expectation), boxed sum. The stochastic version with $k$ samples and the ascent update. Also give $\nabla_{v_c}\ell$ and $\nabla_{v_{w_i}}\ell$. *Intuition:* each term is (label − prediction) × (other vector), as for softmax + cross-entropy in Problem Set 2; $k+1$ dot products instead of $|V|$.
- **Step 3: SGNS approximately factorises a shifted PMI matrix** (Levy & Goldberg, 2014). Corpus counts $\#(w,c)$, $\#(w)$, $\#(c)$, $|D|$; total objective; per-pair objective in $x = v_c^\top v_w$; assumption that $d$ is large enough for every $x$ to be free; solve, check the second derivative, boxed $v_c^\top v_w = \mathrm{PMI}(w,c) - \log k$; $WC^\top = M$.
- **Why only approximate:** $d \ll |V|$ (weighted low-rank approximation, frequent pairs fitted best); unseen pairs have $\mathrm{PMI} = -\infty$; word2vec's smoothed noise, subsampling and dynamic windows change the counts; SGD stops early.
- *Intuition:* PMI is the same kind of log-ratio as the idf of Problem 1; similar PMI rows give similar vectors (distributional hypothesis).

## Verify
- The notebook is valid (`nbformat.validate`), and only the three Problem 1–3 cells changed.
- Every formula renders in Jupyter (no unbalanced `$`, `\boxed` used only in display math).
- Report anything in the PDF statement you had to reinterpret.
- Do not git commit.
