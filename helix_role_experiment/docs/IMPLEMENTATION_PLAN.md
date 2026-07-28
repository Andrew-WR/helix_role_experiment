# Repository implementation plan

## Constraints discovered in the audit

- The checkout has no committed history and all pre-existing files are
  untracked; new work must avoid overwriting them.
- The supplied paper and original experimental traces are absent.
- The local bundled Python has NumPy and pandas but not PyTorch, Transformers,
  SciPy, scikit-learn, matplotlib, or pytest.
- Kaggle/GPU execution is therefore prepared but cannot be run locally without
  installing the optional model dependencies.

## Package boundary

All new code lives under `helix_role_experiment/`. It never imports the router
or its PLL. The scientific core is dependency-light NumPy; model collection
and Parquet/plot conveniences are optional extras with actionable errors.

## Staged implementation

### Stage 0: mathematical and synthetic validation

- typed trace records and deterministic IDs;
- Fourier projection with explicit normalization;
- tautology phase recovery;
- smooth-process null generators;
- trend, window, reflection, and cosine-basis sensitivity;
- Grassmann, generalized-eigen, and complex-coefficient estimators;
- full 2D whitening and radius uncertainty;
- intervention algebra and EOS orthogonalization;
- deterministic controlled tasks and counterfactual validators;
- grouped bootstrap and grouped cross-validation;
- dependency-light SVG figures.

Acceptance: all standard-library `unittest` tests pass with known synthetic
answers and the end-to-end synthetic script produces tables and figures.

### Stage 1: tiny-model smoke

- generic Hugging Face causal-LM adapter with architecture-independent layer
  discovery;
- token-aligned hooks, greedy or sampled decoding, layer selection, EOS logits,
  and resumable NPZ shards;
- optional tiny CPU model config;
- a no-download deterministic synthetic backend for CI/local validation;
- collection, basis fit, observational cross, causal hook, analysis, and plots.

Acceptance: every script completes on the synthetic backend; the Hugging Face
backend completes when optional dependencies/model access are available.

### Stage 2: discovery

- 50 problems per family;
- all-layer observational traces;
- eight crossed variants and five continuations;
- selected layer/window interventions;
- problem-level discovery report and variance estimates.

Acceptance: data-quality gates pass, estimator and layer choices are frozen,
and the confirmatory manifest is generated without test-set inspection.

### Stage 3: confirmatory

- at least 300 problems per family;
- held-out templates and one held-out family;
- 8–16 stochastic continuations;
- base Qwen 27B, fine-tuned checkpoint, and unrelated family;
- pre-registered contrasts and Holm correction;
- content-removal classification and token/accuracy Pareto analysis.

## Script contract

Each numbered script:

1. reads a versioned JSON config;
2. records a config hash, seed, model/tokenizer revisions, environment, and
   deterministic request IDs;
3. writes atomic, resumable shards;
4. raises on unexpected errors and records expected exclusions;
5. produces machine-readable CSV/JSON and optional Parquet;
6. never uses final output length in an online feature.

## Missing information that materially affects execution

The following are represented as required full-run config fields rather than
guessed:

- original paper/PDF and its exact layer/token conventions;
- base Qwen 27B model ID and revision;
- helix-fine-tuned checkpoint ID/revision and training procedure;
- chat template and exact decoding settings used in the prior result;
- datasets, evaluation rubric, and human-evaluation records;
- available GPU type/count, storage budget, and model-access credentials.

The package remains runnable on synthetic and tiny public backends without
these items. Base-versus-fine-tuned conclusions remain blocked until they are
provided.

## Evidence boundary

Passing tests establishes algebra and pipeline integrity only. A tiny-model
smoke run establishes engineering viability only. Scientific claims require
the pre-registered controlled discovery and confirmatory data.

