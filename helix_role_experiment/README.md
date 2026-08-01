# Helix role experiment

This standalone package tests whether a shared low-frequency activation
subspace is semantic progress, generic procedure, sequence position, expected
completion, termination readiness, confidence, a mixture, or an artifact. It
does not import or use the router PLL.

The key scientific design is a problem-level progress–position cross with
teleport, rollback, loop, and answer-permission conflicts, followed by
content-distinct causal interchange under fixed-length and EOS controls.

## Evidence boundary

The bundled synthetic backend has a known latent variable. Its results validate
math, storage, statistics, interventions, and figures; they are **not evidence
about an LLM**. The tiny model is an engineering smoke test. Scientific claims
require the discovery and locked confirmatory studies.

The original paper/PDF, original traces, Qwen 27B model identifier, fine-tuned
checkpoint, decoding settings, and prior evaluation records were not present
in this checkout. Placeholder full-run config fields intentionally fail rather
than silently choosing a different model.

Read first:

- `docs/MATHEMATICAL_AUDIT.md`
- `docs/PREREGISTRATION.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/OUTPUT_SCHEMA.md`
- `docs/FINAL_REPORT_TEMPLATE.md`

## Local synthetic validation

From `helix_role_experiment/`, with any Python containing NumPy:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
python scripts/run_smoke.py --config configs/smoke_synthetic.json
```

In this Codex desktop environment, `python` is not on PATH. The equivalent
verified runtime command is:

```powershell
$py = 'C:\Users\hp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = "$PWD\src"
& $py -m unittest discover -s tests -v
& $py scripts/run_smoke.py --config configs/smoke_synthetic.json
```

The seven explicit stages are:

```bash
python scripts/01_collect_traces.py --config configs/smoke_synthetic.json
python scripts/02_fourier_audit.py --config configs/smoke_synthetic.json
python scripts/03_fit_shared_subspace.py --config configs/smoke_synthetic.json
python scripts/04_build_counterfactual_prefixes.py --config configs/smoke_synthetic.json
python scripts/05_run_observational_cross.py --config configs/smoke_synthetic.json
python scripts/06_run_causal_interventions.py --config configs/smoke_synthetic.json
python scripts/07_analyze_results.py --config configs/smoke_synthetic.json
```

## Sentence-level next-subgoal readiness experiment

The resource-limited cross-domain pilot uses 50 exactly scoreable MATH-500
problems and 50 HumanEval problems. SWE-EVO is intentionally deferred: its 48
repository-scale tasks require an interactive OpenHands/SWE-agent scaffold and
hundreds of tests, so treating it as one-shot generation would not be a valid
evaluation. HumanEval lets activation generation remain asynchronous from code
execution and tests whether the signal extends beyond mathematics.

The design keeps expensive model work separate from labeling and probe fitting:

```bash
python scripts/07a_collect_sentence_activations.py --config configs/qwen_9b_readiness_kaggle.json
python scripts/07b_label_subgoal_events.py --config configs/qwen_9b_readiness_kaggle.json prepare
python scripts/07b_label_subgoal_events.py --config configs/qwen_9b_readiness_kaggle.json run --concurrency 8
python scripts/07c_fit_survival_probes.py --config configs/qwen_9b_readiness_kaggle.json
python scripts/07d_run_readiness_steering.py --config configs/qwen_9b_readiness_kaggle.json
```

Stages 07a and 07d launch two child processes with `CUDA_VISIBLE_DEVICES=0`
and `1`; each process loads one 4-bit Qwen3.5-9B replica and decodes batches of
two. Stage 07a stores activations only at locally parsed sentence boundaries.
The LaTeX-aware scanner protects decimals, abbreviations, inline/fenced code,
math delimiters, and environments.

Stage 07b obtains `OPENAI_API_KEY` from the environment, or from the Kaggle
secret of the same name, and never writes the key. It sends concurrent,
immediate Responses API calls with strict Structured Outputs. Valid results
from the original whole-trajectory labeler are preserved without rewriting.
Only missing trajectories are divided into chunks of at most 24 target
sentences; Luna receives the complete trajectory as read-only context, while
the schema permits only the current chunk's exact IDs and count. Each chunk is
saved separately and merged into the original full-result format only after
all chunks validate, so an interrupted run resumes without paying to relabel
finished work. Exact-evidence and consistency validation applies per chunk and
again after merging, with up to three corrective retries. A conservative
pre-run estimate is checked against the configured USD 5.60 hard guard, and
actual token usage/cost is written after the run.

Generated code is never run during model collection or steering. Stage 07d
exports one `humaneval_<condition>.jsonl` file per condition. Evaluate these
later in a disposable, network-disabled container or VM using the official
[OpenAI HumanEval evaluator](https://github.com/openai/human-eval); its own
README warns that model-generated code is untrusted:

```bash
evaluate_functional_correctness /kaggle/working/qwen9b_readiness/tables/humaneval_baseline.jsonl
evaluate_functional_correctness /kaggle/working/qwen9b_readiness/tables/humaneval_gated.jsonl
evaluate_functional_correctness /kaggle/working/qwen9b_readiness/tables/humaneval_always.jsonl
evaluate_functional_correctness /kaggle/working/qwen9b_readiness/tables/humaneval_random.jsonl
python scripts/07e_finalize_readiness_results.py --config configs/qwen_9b_readiness_kaggle.json
```

The primary gate passes only if the same gated method generalizes across math
and code and achieves either at least 10% fewer output tokens with no more than
one percentage point accuracy loss, or at least five percentage points higher
accuracy without increasing mean output tokens. Always-on and random-direction
conditions are causal controls, not candidate products. This first model can
only pass `within_model_success`. Re-run the same frozen method with a second
open model and pass its `readiness_success_gates.json` via
`--replication-gates`; `commercial_success` remains false until the same
condition passes both models.

All output paths are controlled by `output.root`. Trace shards are resumable:
existing deterministic request IDs are not rewritten.

## Tiny Hugging Face model

Install optional dependencies in a GPU environment:

```bash
pip install -e '.[model,analysis,test]'
python scripts/01_collect_traces.py --config configs/tiny_model_smoke.json
python scripts/02_fourier_audit.py --config configs/tiny_model_smoke.json
python scripts/03_fit_shared_subspace.py --config configs/tiny_model_smoke.json
python scripts/04_build_counterfactual_prefixes.py --config configs/tiny_model_smoke.json
python scripts/05_run_observational_cross.py --config configs/tiny_model_smoke.json
```

The generic collector supports Qwen, Llama, GPT-2-style, GPT-NeoX, and
decoder-layer layouts; it records the hidden activation used to predict each
aligned generated token. EOS-disabled fixed-token generation and hook-time
intervention callbacks are implemented. Stage 6 uses a fixed one-token
continuation budget and scores the target task's valid next-transition tokens
at the donor's abstract stage; EOS probability is a separate diagnostic and
output length is never substituted for task state.

## Kaggle: clone versus attached code dataset

Clone the repository into `/kaggle/working` while developing. An attached
Kaggle dataset is a static versioned snapshot; it does not update when GitHub
changes. Cloning fetches the current branch, while checking out a recorded
commit gives the same reproducibility as a dataset snapshot.

With Kaggle Internet enabled:

```bash
cd /kaggle/working
git clone --depth 1 \
  https://github.com/Andrew-WR/helix_role_experiment.git \
  helix-role-src
cd /kaggle/working/helix-role-src/helix_role_experiment
python -m pip install -q -U -e '.[model,analysis]'
git rev-parse HEAD
```

Qwen3.5 is not recognized by Kaggle's older preinstalled Transformers build.
The model extra therefore requires `transformers>=5.14.1,<6` and
`peft>=0.18`. If either package was already imported in the notebook process,
restart the Kaggle session after installation; commands launched as a new
`!python` process will see the upgraded packages directly.

On a restarted session, clone again. During one live session, update with:

```bash
git -C /kaggle/working/helix-role-src pull --ff-only
```

Keep model and LoRA weights as Kaggle inputs. The Qwen configs use:

```text
/kaggle/input/models/andrewwafik/turbo-qwen-27b/pytorch/human_eval_200/1/checkpoint-step-200
```

and automatically fall back to:

```text
/kaggle/input/models/andrewwrufail/turbo-qwen-27b/pytorch/default/1/checkpoint-step-200
```

The loader selects the first configured directory containing
`adapter_config.json`, resolves its
`base_model_name_or_path`, infers the adapted transformer layer from
`layers_to_transform` or adapter weight keys, loads the base in 4-bit NF4
across both T4s, and optionally enables the adapter. If the base repository is
private, authenticate to Hugging Face before preflight.

Run the fail-fast check before loading weights:

```bash
python scripts/00_kaggle_preflight.py \
  --config configs/qwen_27b_kaggle_smoke.json
```

Then run the LoRA smoke:

```bash
python scripts/01_collect_traces.py \
  --config configs/qwen_27b_kaggle_smoke.json
python scripts/02_fourier_audit.py \
  --config configs/qwen_27b_kaggle_smoke.json \
  --layers 32-63
python scripts/03_fit_shared_subspace.py \
  --config configs/qwen_27b_kaggle_smoke.json \
  --layers 32-63
python scripts/04_build_counterfactual_prefixes.py \
  --config configs/qwen_27b_kaggle_smoke.json
python scripts/05_run_observational_cross.py \
  --config configs/qwen_27b_kaggle_smoke.json
python scripts/05b_compare_open_progress_models.py \
  --config configs/qwen_27b_kaggle_smoke.json \
  --layers late-half
python scripts/06b_falsify_generalized_helix.py \
  --config configs/qwen_27b_kaggle_smoke.json \
  --layers 51,55,59 \
  --pairs-per-family 2
python scripts/06c_behavioral_helix_interventions.py \
  --config configs/qwen_9b_base_kaggle_smoke.json \
  --layer 31 \
  --math500-level 1 \
  --generation-safety-ceiling 8192 \
  --progress-step 0.05 \
  --transport-alpha 1.0 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20
python scripts/06_run_causal_interventions.py \
  --config configs/qwen_27b_kaggle_smoke.json
python scripts/07_analyze_results.py \
  --config configs/qwen_27b_kaggle_smoke.json
```

Run `qwen_27b_base_kaggle_smoke.json` identically for the adapter-disabled
baseline. Both configs use the adapter metadata to resolve the identical base
checkpoint.

Before causal interventions, `05b_compare_open_progress_models.py` tests
whether the fixed closed Fourier `k=1` assumption is actually preferred. In
one file it:

- compares closed `k=1` output-token trajectories with open linear,
  polynomial, DCT, spline, and drift-plus-rotation bases using contiguous
  held-out token blocks;
- compares raw controlled-prefix activations under position/confidence/EOS
  nuisances plus linear, open-curve, closed-loop, and spiral-with-drift
  progress models using problem-grouped and leave-family-out evaluation;
- obtains actual formatted Qwen tokenizer lengths rather than treating
  whitespace word counts as token counts; and
- reports off-plane residual norm divided by centered activation norm, making
  manifold distance comparable across depth.

It writes `temporal_open_basis_*`, `progress_manifold_model_*`,
`observational_geometry_normalized.csv`, and
`counterfactual_actual_token_counts.csv` under `tables/`, plus figures 14 and
15. A positive `incremental_r2_vs_nuisance` means structural progress adds
held-out activation information after actual token position, confidence,
operation, EOS logit, and termination are included. The default
`--layers late-half` follows the resource-limited late-layer plan; use an
explicit range such as `--layers 32-63` to lock it.

`06b_falsify_generalized_helix.py` is a deliberately small, falsification-first
causal test. It fits a nuisance-residualized minimal generalized helix

```text
h_f(s) = mu + s*u + r_f(s)[cos(omega_f*s)*v_f + sin(omega_f*s)*w_f]
```

where `r_f(s)` is constrained to a positive linear function. Phase span and
radius slope are selected from ordinary trajectories by leave-one-problem-out
prediction. Each intervention geometry is then refit with the target problem
excluded. Endpoint transfers compare the full helix with equal-norm linear,
nested linear-plus-family-specific-closed-`k=1`, wrong-family, reversed,
off-model random, and EOS-orthogonal directions, together with rotational
ablation and a 0.5x/1x/1.5x dose response.

Run it on only a small layer set frozen before causal outcomes are inspected.
It writes `generalized_helix_causal_outcomes.csv`,
`generalized_helix_intervention_summary.csv`,
`generalized_helix_falsification_gates.csv`,
`generalized_helix_geometry_selection.csv`, and
`generalized_helix_model_fit.csv`. A failed smoke gate is a reason to reject or
simplify that part of the model, not a p-value; discovery-scale replication
remains problem-grouped.

`06c_behavioral_helix_interventions.py` is the compact behavioral causal test
and is self-contained: files 01-05, 05b, and 06b do not need to be run. It
selects two fixed, disjoint, level-1 integer-answer MATH-500 problems from the
study seed and writes their identities before generation. The first is the
calibration problem; the second is the test problem. Neither is silently
replaced if Qwen fails it.

The resource-limited default uses the adapter-disabled
`Qwen/Qwen3.5-9B` checkpoint through
`configs/qwen_9b_base_kaggle_smoke.json` and tests its last text layer, 31.
The existing Qwen3.6-27B LoRA is architecture-specific and must not be attached
to this 9B checkpoint.

The calibration run is a complete natural Qwen reasoning rollout. File 06c
records the requested layer at every generated output token and defines
calibration progress as `t/(T-1)`. It fits linear, linear-plus-closed-`k=1`,
and generalized-helix curves directly to this trajectory. Generalized-helix
phase span and radius slope are selected using contiguous blocked temporal
validation, not random token splits. The unmodified test rollout then measures
centered out-of-prompt transfer before causal results are interpreted.

The causal operation is local forward transport throughout reasoning:

```text
delta_t = h(min(s_t + 0.05, 1)) - h(s_t)
s_t = min(t / T_baseline, 1)
```

The generalized-helix, linear, and linear-plus-closed-`k=1` local steps are
norm matched at every token. This is important: unlike the old endpoint chord,
the local closed-`k=1` displacement is nonzero and therefore remains a real
rotational comparator. All three interventions reuse the baseline sampling
seed.

The dedicated semantic-progress config enables Qwen thinking mode. The
authored-subgoal labels are applied to the resulting reasoning trace.

There is no ordinary reasoning-token budget. Generation stops at a complete
`FINAL:` line or EOS. `--generation-safety-ceiling 8192` is only an emergency
loop guard. Both fixed baselines must emit a correct final answer. Final-answer
normalization strips decoded special tokens such as `<|im_end|>`.

After generation, Qwen is released and `Qwen/Qwen3-Embedding-0.6B` is loaded
for outcome-only semantic analysis. A LaTeX-aware sentence scanner protects
decimals, math delimiters, environments, and code. Each sentence receives
auditable multi-label reasoning categories using lexical evidence plus
category-prototype similarity. A sentence is redundant when its maximum cosine
similarity to any earlier sentence exceeds 0.65; sensitivity results are also
reported at 0.60, 0.70, and 0.75.

The primary efficiency result is the percentage output-token delta from the
correct baseline; negative is favorable only when correctness is preserved.
The script also reports reasoning-token deltas, correct-answer sequence log
odds, category timing, stage transitions, stage revisits, and redundant
sentence/token fractions. This two-prompt run is a case study, not a
population-level estimate.

Outputs include:

- `behavioral_helix_two_prompt_design.csv`
- `behavioral_helix_calibration_trace.csv`
- `behavioral_helix_natural_traces_layer_<LAYER>.npz`
- `behavioral_helix_trajectory_model_fit.csv`
- `behavioral_helix_test_transfer.csv`
- `behavioral_helix_outcomes.csv`
- `behavioral_helix_key_results.csv`
- `behavioral_helix_sentence_audit.csv`
- `behavioral_helix_semantic_summary.csv`

`06d_semantic_progress_interventions.py` is the replacement, falsification-first
semantic experiment. It is also self-contained and uses
`data/simple_multistep_math_dataset_latex.json`. These are deliberately easy,
LaTeX-formatted problems with prescribed three- or four-step paths and
conservative automatic grading. By default, problem 1 is the calibration trace
and problem 2 is the held-out causal test. The other eight problems remain untouched for a later
frozen replication. All candidate layers are captured in the same two baseline
generations, so testing five late
layers does not require five rollouts.

The first pair uses the same prescribed substitution template with different
coefficients. It tests within-template transfer under tightly matched
difficulty; it does not establish cross-method semantic generalization. Problems
3 through 10 provide the later cross-method test after the pilot is frozen.

The semantic threshold is not fixed at 0.65. Qwen3-Embedding-0.6B embeds the
minimal authored steps and each problem's threshold is its own mean distinct
pairwise cosine similarity, frozen across that problem's conditions before any
outcome comparison. A generated sentence advances progress
only if it matches the next ordered authored subgoal above that frozen
threshold, with at most one subgoal credited per sentence. All `FINAL:` spans
are excluded from progress and recurrence labels. Progress is a staircase that
changes after sentence completion; it is not retrospectively ramped across the
sentence. Activation row `t` is the predictive state for output token `t`, so
its label is the number of subgoals completed by the preceding output prefix.
Recomputation means re-hitting an already completed subgoal without
advancing the ordered frontier. The full sentence-by-step similarity matrix is
exported for audit.

At each candidate layer, the script fits activation variation attributable to
authored subgoal progress after controlling for normalized output position and
closed `k=1` sine/cosine terms. It keeps this conditional coefficient rather
than post-hoc orthogonalizing it. Layer selection uses contiguous blocked-CV
incremental R2 on the calibration trace. The frozen direction must show both
partial-correlation and event-locked movement at held-out subgoal boundaries
before its causal result is taken seriously.

The causal edit is a four-token pulse immediately after held-out baseline
subgoal 1, not persistent steering throughout generation. Semantic-forward,
semantic-reverse, linear, and closed-`k=1` pulses are norm matched to half the
calibration layer's median native token-to-token activation displacement. Every
condition uses the held-out baseline clock `t/T_baseline`. Correctness and
ordered-subgoal coverage are hard gates before shorter output or earlier
progress AUC can count as success. Seven explicit falsification gates are written;
this two-problem run remains a pilot rather than a population claim.

The script explicitly applies Qwen's chat template with `enable_thinking=true`.
Thinking tags are expected and are recorded rather than treated as an invalid
run. The prompt requires one mathematical operation per sentence and an exact
final line of `FINAL: <answer>`. Generation ends at that line or EOS; 8192
tokens is only an emergency ceiling.
Files 01 through 06c are not prerequisites. A run manifest prevents a changed
design from overwriting results in the same output root.

Run:

```bash
python scripts/06d_semantic_progress_interventions.py \
  --config configs/qwen_9b_semantic_progress_kaggle.json
```

The concise files to inspect first are:

- `semantic_key_results.csv`
- `semantic_falsification_gates.csv`
- `semantic_intervention_outcomes.csv`

For auditing the labels and signal fit, inspect:

- `semantic_subgoal_threshold.csv`
- `semantic_subgoal_alignment.csv`
- `semantic_signal_layer_selection.csv`
- `semantic_event_locked_summary.csv`
- `semantic_run_manifest.json`

The smoke and discovery configs use `adapter_neighborhood`: the adapted block
plus two blocks on either side, together with five depth sentinels. This keeps
T4 runtime and disk use tractable while still measuring upstream and
downstream changes. Confirmatory configs retain `layers: all` and require a
larger storage/compute budget.

## Discovery and confirmatory Qwen 27B runs

The supplied Kaggle configs resolve the base model from the PEFT adapter
metadata. Pilot and freeze the Stage 6 first-transition verbalizations;
optionally upgrade them to multi-token transition likelihoods before locking
the confirmatory run.

Discovery:

```bash
python scripts/00_kaggle_preflight.py --config configs/qwen_27b_discovery.json
python scripts/01_collect_traces.py --config configs/qwen_27b_discovery.json
python scripts/02_fourier_audit.py --config configs/qwen_27b_discovery.json
python scripts/03_fit_shared_subspace.py --config configs/qwen_27b_discovery.json
python scripts/04_build_counterfactual_prefixes.py --config configs/qwen_27b_discovery.json
python scripts/05_run_observational_cross.py --config configs/qwen_27b_discovery.json
python scripts/06_run_causal_interventions.py --config configs/qwen_27b_discovery.json
python scripts/07_analyze_results.py --config configs/qwen_27b_discovery.json
```

This is the LoRA condition. Repeat with
`configs/qwen_27b_base_discovery.json` for the adapter-disabled base condition.
Confirmatory uses `qwen_27b_confirmatory.json` and
`qwen_27b_base_confirmatory.json` once the pre-registration, estimator, layers,
scoring rules, and sample sizes are locked.

Run base, fine-tuned, and unrelated-family models into separate output roots.
Never pool their basis calibration. Compare them only after within-model frozen
bases and paired problem IDs are established.

Natural external-validity tasks can be added through `tasks.natural_jsonl`.
Each JSONL row must contain `problem_id`, `prompt`, `reference_answer`, and
`task_family`; these traces are observational and do not replace exact-state
controlled tests.

After objectively scoring paired base and fine-tuned generation JSONL files
(`problem_id`, `text`, `correct`, `token_count`), run:

```bash
python scripts/08_compare_base_finetuned.py \
  --base-output /path/to/base/results \
  --tuned-output /path/to/tuned/results \
  --base-generations /path/to/base_generations.jsonl \
  --tuned-generations /path/to/tuned_generations.jsonl \
  --out /path/to/paired_comparison
```

This writes plane alignment, paired token/accuracy changes, removed-content
categories, and Figures 12–13. The deterministic sentence classifier is an
auditable baseline; confirmatory content labels should be blinded and
verifier- or human-validated.

## Implementation notes

- Complete-output centering and final length are confined to retrospective
  spectral audit and shared-basis calibration.
- Per-trace PCA axes are never averaged.
- Phase is excluded when calibration-defined radius is too small; raw
  coordinates remain available.
- Exceptions are raised; expected exclusions receive explicit reason codes.
- Splits, request IDs, configs, environment, model revisions, and tokenizer
  revisions are recorded.
- Figures are dependency-light SVG and show missing fine-tuned evidence
  explicitly instead of fabricating panels.
