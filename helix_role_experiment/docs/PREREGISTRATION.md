# Pre-registration: functional role of the candidate low-frequency subspace

## Status

This document is written before inspecting results from the new controlled
experiment. Synthetic unit-test outputs and tiny-model engineering smoke tests
are not scientific evidence and are excluded from confirmatory inference.

## Primary question and estimand

Does a frozen activation-space subspace carry and causally implement semantic
advancement through a task, rather than sequence position, generic procedure,
expected completion, termination readiness, confidence, or analysis artifact?

The primary estimand is the paired, problem-level average treatment effect of
candidate-subspace interchange on the probability of a valid next
computational transition under fixed-length, EOS-neutralized decoding:

\[
\tau_{\mathrm{semantic}} =
E_p[\Pr(\text{source-progress-consistent next state}\mid do(z_t=z_s))
-\Pr(\cdot\mid\text{norm-matched control})].
\]

Co-primary descriptive estimands are matched counterfactual changes
\(\Delta_{\text{progress}}\), \(\Delta_{\text{position}}\), and
\(\Delta_{\text{termination}}\) in the two raw whitened coordinates, radius,
and reliable phase.

## Hypotheses

- H1: semantic task progress.
- H2: generic procedural phase.
- H3: absolute/relative sequence clock or iteration count.
- H4: expected or retrospective sequence completion.
- H5: termination or answer-emission readiness.
- H6: confidence or solution commitment.
- H7: a mixture with separable unique effects.
- H8: Fourier, endpoint, positional, lexical, formatting, or aggregation
  artifact.

No hypothesis is designated as the favored outcome.

## Design

### Units and splits

The independent unit is a generated problem instance. All prefixes, tokens,
continuations, and interventions from one problem remain in one split.
Template and entity vocabularies are also held out for confirmatory tests.

Splits are deterministic from `(study_seed, problem_id)`:

- 50% shared-plane calibration;
- 20% probe/hyperparameter training;
- 10% validation and regularization selection;
- 20% locked test.

The confirmatory run additionally holds out complete task templates and at
least one task family for cross-family transfer.

### Controlled families

1. Iterative state machines: parity, modular polynomial iteration, finite-state
   machines, stack/register programs, and pointer chasing.
2. Fictional ontologies with invented entities, positive/negative relations,
   and distractors.
3. Dependency graphs with alternative valid orders, invalid commitments, and
   explicit rollback.

Every prefix stores exact state, goal distance, completed dependencies,
invalid commitments, and valid next transitions. Structural progress and
behavioral competence are separate variables.

### Crossed manipulations

Each problem attempts:

- same state / different length: concise, paraphrase, redundant-valid,
  repeated-summary, confirmation, and plausible digression;
- same length / different state: productive, irrelevant, repeated, wrong
  branch, corrected/uncorrected, and supplied lemma;
- teleport, rollback, stall, and loop;
- answer-known/work-unfinished, work-complete/answer-unknown, confident guess,
  and complete-but-answer-forbidden;
- matched planning, calculation, uncertainty, backtracking, checking,
  consolidation, and final-emission operations.

Automatic validators reject variants that do not preserve their declared exact
state or violate the requested transition. Token-count matching tolerance is a
configuration field and is reported.

## Outcomes

### Primary causal outcomes

1. Valid-next-state probability under a fixed continuation budget with EOS
   disabled.
2. Causal-abstraction interchange accuracy on content-distinct source/target
   pairs.
3. Next-operation category shift in the pre-registered source-consistent
   direction.

### Secondary causal outcomes

- fixed-budget correctness;
- verification, correction, or backtrack probability;
- answer-versus-continue choice;
- persistence after intervention removal;
- downstream KL and activation-norm change;
- output length and EOS probability, treated as termination diagnostics.

### Observational outcomes

- held-out joint Gaussian likelihood/RMSE of whitened 2D coordinates;
- circular resultant error only where radius is reliable;
- grouped cross-validated incremental \(R^2\);
- problem-level paired effect sizes;
- held-out first-harmonic signal-to-residual energy;
- conditional information estimates used only when permutation calibration is
  valid.

Predictor blocks are position, retrospective completion, termination,
procedure, confidence, semantic state, and their pre-specified mixed model.

## Basis estimation

Only calibration problems fit the basis. Three frozen estimators are compared:

1. Grassmann/projector mean;
2. pooled spectral generalized eigenvectors, with ridge selected on validation
   data;
3. complex first-harmonic low-rank SVD with complex scalar gauge.

Estimator selection uses validation spectral selectivity, not semantic test
labels. Final two-dimensional covariance is fully whitened. Random-plane
baselines are dimension- and layer-matched.

Phase is undefined when whitened radius is below the calibration-set quantile
specified by `analysis.radius_quantile` (default 0.10). Such rows remain in
coordinate analyses but are excluded from phase summaries with exclusion rates
reported.

## Interventions and controls

Candidate-plane rotation, donor-coordinate transplant, and full-frame
interchange are applied by layer and temporal window. Controls are:

- norm-matched random directions and random planes;
- shuffled or position-only donors;
- positional and EOS directions;
- direct EOS bias and brevity prompts;
- residual, plane-orthogonal, magnitude-only, phase-only, and radial changes.

Intervention norms are matched at the application point. Every result reports
downstream KL and hidden-state norm change.

## EOS exclusion tests

At each layer, the candidate plane is optionally orthogonalized to the local
EOS-logit gradient. Causal conclusions beyond termination require effects on
valid next transitions or fixed-budget correctness in both:

1. fixed-token continuation with EOS disabled; and
2. intervention/control pairs matched on EOS logit within the configured
   tolerance.

A length effect alone, or an effect disappearing under both controls, is
classified as termination control.

## Statistical analysis

- All confidence intervals resample problems, stratified by task family.
- Primary paired contrasts use two-sided 95% bootstrap intervals and report
  standardized and raw effects.
- Nested predictor blocks are evaluated with grouped folds.
- Wrapped phase is modeled as sine/cosine coordinates, not scalar OLS.
- Mixed-effects inference, when the optional dependency is available, includes
  problem and family intercepts; grouped bootstrap remains the dependency-light
  reference analysis.
- Confirmatory multiplicity is controlled across the three primary causal
  outcomes with Holm correction. Layer scans are exploratory unless a layer
  window was chosen on calibration/validation only.
- Effect sizes and intervals are primary; p-values are supplementary.

## Exclusions

Pre-specified exclusions:

- trace shorter than 8 generated tokens for first-harmonic analysis;
- missing or nonfinite activations;
- hook/token misalignment;
- counterfactual failing exact-state validation;
- truncation before the required outcome;
- low-radius phase rows, for phase outcomes only;
- intervention norm outside 10% of its paired control;
- EOS-logit match outside configured tolerance, for EOS-matched analysis.

Every exclusion is recorded with a machine-readable reason. No exception is
silently swallowed. Problems are not excluded for being incorrect; correctness
is a factor and pre-specified subgroup.

## Sample sizes

- Stage 1 engineering smoke: 6–12 problems total; no inference.
- Stage 2 discovery: 50 problems per family, about 8 variants, 5 continuations.
- Stage 3 confirmatory: at least 300 problems per family, 8–16 continuations,
  base and helix-fine-tuned Qwen 27B, plus an unrelated model family if
  feasible.

Power simulation uses discovery-level problem variance and is frozen before
confirmatory unblinding.

## Interpretation rules

“Semantic-progress representation” is permitted only if all six hold:

1. invariance across concise, verbose, paraphrased, and padded prefixes;
2. teleport advance, loop stall, and rollback regression/reset;
3. unique held-out semantic contribution after position/completion controls;
4. content-preserving cross-task causal interchange;
5. fixed-length and EOS-matched effects on reasoning-state transitions;
6. held-out cross-family robustness.

Otherwise:

- padding advance without rollback response: sequence clock;
- dominant sampled-remaining-length prediction: completion estimator;
- answer/EOS effects abolished by EOS control: termination controller;
- operation-category transfer without exact-distance transfer: procedure;
- confidence sensitivity dominating remaining work: commitment/confidence;
- decoding without causal effect: diagnostic correlate;
- effects only on fixed iterative tasks: narrow algorithmic controller;
- no held-out plane above smooth-process nulls: artifact;
- multiple unique observational and causal effects: mixed representation.

