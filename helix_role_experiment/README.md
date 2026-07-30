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
  --config configs/qwen_27b_kaggle_smoke.json \
  --layer 63 \
  --problems 1 \
  --rollouts 1 \
  --benchmark math500 \
  --math500-levels 2,3 \
  --calibration-problems 4 \
  --controls core \
  --generation-safety-ceiling 8192 \
  --pulse-tokens 16 \
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
that replaces the one-token assay for substantive conclusions. It is
self-contained: files 01-05, 05b, and 06b do not need to be run. After loading
the model once, file 06c generates four small modular-arithmetic calibration
problems, records only their concise state trajectories at the requested layer,
and fits the generalized helix directly in raw activation space. This compact
calibration is cached as
`behavioral_helix_internal_calibration_layer_<LAYER>.npz`; reruns in the same
Kaggle environment reuse it automatically. The cache is updated atomically
after every one-token probe, so an interrupted calibration resumes at the next
state rather than starting over. Use
`--rebuild-internal-calibration` only when intentionally replacing that cache.
It then adapts the
counterfactual-rollout principle from *Thought Anchors*: apply a brief
activation pulse (16 tokens by default), remove it, and measure its downstream
influence on the reasoning trace and final answer. It uses two contexts:

- acceleration on a held-out MATH-500 problem from its initial state, where
  the strongest result is a correct `FINAL:` answer in fewer reasoning
  tokens; and
- repair of a planted wrong commitment, where useful backward steering must
  induce planning, uncertainty management, checking, or repetition **and**
  improve final correctness. Slower or longer text alone does not pass.

The core comparison retains baseline, desired generalized-helix, linear, and
linear-plus-family-closed-`k=1` conditions.
Complete correct-versus-wrong continuation log odds replace the saturated
first-token probability. A direct answer-logit direction is included only as a
positive control: if it cannot move sequence log odds, all behavioral gates are
reported as assay-inconclusive rather than as evidence against the helix.

Qwen thinking mode is enabled explicitly through its chat template. Generation
defaults follow the model's recommended thinking-mode sampling settings
(`temperature=1.0`, `top_p=0.95`, `top_k=20`). The default behavioral task is a
deterministically selected integer-answer MATH-500 problem at level 2 or 3.
The iterative-state helix is fitted only on the existing controlled
calibration prompts generated inside file 06c, so MATH-500 is a held-out
cross-task transfer test rather than geometry training data.

There is no ordinary reasoning-token budget. Generation stops at a complete
`FINAL:` line or EOS. `--generation-safety-ceiling 8192` is only an emergency
loop guard; reaching it invalidates the prompt instead of being treated as a
causal result. Before scoring any intervention, the script runs every
unmodified baseline rollout. A failure writes
`behavioral_helix_baseline_pilot.csv` and aborts. The default `--controls core`
keeps only baseline, generalized helix, linear, and linear-plus-closed-`k=1`;
use `--controls full` to add opposite and random controls. Core mode cuts the
expensive generations from 12 to 8 per problem.

The script writes only `behavioral_helix_outcomes.csv` for auditability and
`behavioral_helix_key_results.csv` for interpretation. Output length is a weak
diagnostic; the primary gates are faster correct completion and
slow-reflection-assisted repair.

Generated activations and per-token EOS logits are not copied back to CPU, and
generation stops as soon as a nonempty `FINAL:` line appears. For the fastest
plumbing check, keep one MATH-500 problem, one rollout, and core controls as in
the command above. Increase problems and rollouts only after baseline
completion and correctness are calibrated.

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
