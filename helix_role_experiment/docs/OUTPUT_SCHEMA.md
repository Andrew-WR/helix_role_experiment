# Output schema

All tables are UTF-8 CSV with headers. Nested metadata in JSONL is JSON. Array
shards are compressed NPZ with `allow_pickle=False`. The schema version is
stored in `environment.json`.

## `traces/manifest.jsonl`

One row per problem, condition, and recorded layer.

| Field | Type | Meaning |
|---|---:|---|
| `request_id` | string | Deterministic config/problem/condition/layer ID |
| `problem_id` | string | Independent bootstrap/split unit |
| `task_family` | string | Controlled or natural family |
| `condition` | string | Normal or counterfactual condition |
| `split` | enum | calibration/train/validation/test |
| `layer` | integer | Zero-based decoder layer |
| `prompt_token_count` | integer | Tokenized prefix length |
| `token_ids`, `tokens` | arrays | Generated-token alignment |
| `activation_file` | string | NPZ filename relative to trace directory |
| `generated_token_count` | integer | First activation dimension |
| `reached_eos`, `truncated` | booleans | Termination diagnostics |
| `model_id`, revisions | strings | Immutable provenance when supplied |
| `seed` | integer | Request seed |
| `state_ids` | string array | Exact token-aligned computational state |
| `structural_progress` | float array | `1 - remaining/initial` |
| `remaining_distance` | float array | Exact shortest remaining work |
| `operation` | string array | Procedure category |
| `confidence` | float array | Separate commitment estimate |
| `eos_logit` | float array | Token-aligned EOS logit |
| `termination_allowed` | bool array | Instruction-level EOS affordance |
| `exclusion_reason` | nullable string | Pre-specified exclusion |

Each trace NPZ contains `activations: float32[tokens, hidden]`.

## Audit tables

- `tautology_audit.csv`: slope, intercept, unwrapped `r2`, circular MAE,
  circular resultant, singular values, and relative eigengap.
- `spectral_null_audit.csv`: request/problem/layer, null name, draw, `e_k`,
  total non-DC energy, concentration, and residual energy.
- `endpoint_preprocessing_sensitivity.csv`: endpoint distance and the same
  energies for raw, detrended, windowed, DCT, and reflected treatments.
- `projector_similarity.csv`: ordered problem-pair projector similarity by
  layer; square matrices are also stored as NPY.

## Subspace artifacts

`models/subspace_layer_{layer}_{estimator}.npz` contains:

- `basis: float64[hidden,2]`, orthonormal;
- `whitener: float64[2,2]`, full inverse covariance square root;
- `center: float64[hidden]`, calibration-only mean;
- `radius_threshold: scalar`, calibration-only uncertainty cutoff.

`shared_subspace_evaluation.csv` records validation and test selectivity,
random-plane draws, regularization, trace counts, complex component fractions,
wall time, algorithm, CG iterations, and the dense covariance bytes avoided.
`subspace_index.json` records validation-only selection.

## Counterfactual and observational tables

`counterfactual_prefixes.jsonl` stores the exact declared state, state-parent,
condition, operation, confidence, termination permission, token proxy,
validator outcome, and exclusion.

`observational_cross.csv` stores each frozen-plane coordinate, radius, angle,
reliability, manifold distance, competing variables, and one-hot operation
fields. `counterfactual_activations.npz` stores aligned full activations,
variant IDs, and layers for causal pairing.

## Causal table

`causal_interventions.csv` stores source/target problem and progress, control
type, desired and observed behavior shift, causal-abstraction score,
fixed-length/EOS status, EOS overlap/change, downstream KL, intervention norm,
and before/after activation norms.

Scientific full runs must add:

- continuation replicate ID;
- exact valid-next-state probability;
- next-operation distribution;
- correctness and fixed budget;
- actual downstream KL, not the synthetic proxy;
- EOS-match pass/fail and exclusion;
- intervention layer window and token window.

## Statistical tables

- `paired_counterfactual_effects.csv`: problem-level estimate and bootstrap CI
  for progress, position, and termination contrasts.
- `predictor_block_comparison.csv`: grouped-CV likelihood, MSE, total and
  incremental R² in the pre-registered block order.
