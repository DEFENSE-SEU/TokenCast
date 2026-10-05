# TokenCast case-study implementation

This module implements segment-based forecasting and the four-position
Self-Prediction comparison. It uses its own training and checkpoint format.

## Method contract

| Manuscript quantity or operation | Implementation |
| --- | --- |
| Logical-call cost, including accounted retries | `case_study/segments.py`: completed attempt usage aggregated within each logical call |
| Incoming request length | First attempt input usage is a supervision label; its visibility mask remains false unless a causal observation establishes it |
| Segment `(n, g, b)` | `n` counts logical calls; `g = L_out - L_in`; `b = C - n L_in` |
| Composition | `estimators/composable_segment.py`: `(nA+nB, gA+gB, bA+bB+nB*gA)` |
| Shared horizon-conditioned distribution | Shared encoder, count distribution, stopping distribution, conditional continuous coordinates and ending state |
| Conditional continuation | Second segment receives sampled ending state and outgoing length; spend and allowance update arithmetically |
| Point-specific observations | Available at the real cutoff; removed when constructing a future segment |
| Training | Supervised likelihood plus normalized sample-based energy consistency; normalization fitted on training tasks |
| Four predictions | Full-task total, full-call cost, prefix-conditioned full-call cost, remaining-task cost |
| Confirmed accounting | Enforced as lower bounds; terminal scopes use exact recorded accounting |
| Intervals | Raw 5th/95th sample quantiles; frozen calibration fitted separately at each prediction point |

The saved `SegmentConfig` and `fit_report` record the density family and
optimization settings. Task and observed model/harness configuration, including
initial execution limits and the token-limit enforcement policy, use fixed
hashing; the state is a compact causal summary. These choices require empirical
evaluation before paper claims about representation quality or generalization
can be made.

The physical-length density is zero-inflated lognormal. Its log-space standard
deviation is limited to 2.0; the likelihood and sampler share this parameter
constraint. Physical samples are not clipped. Energy distances use float64 and
an off-diagonal U-statistic. Discrete samples use a biased straight-through
gradient approximation. The U-statistic does not make that gradient unbiased.

### Scaling and checkpoint selection

Training uses `condition_scaling="log1p"`. The eight binary progress flags use
`ending_state_distribution="bernoulli"`, with matching binary samples.
Old checkpoints retain linear scaling and Gaussian states when loaded.

Model selection uses the predeclared joint validation likelihood.
`diagnose_composable_bias.py` runs a controlled scaling comparison;
`compare_previous_case_model.py` fits independent tree heads with the same
task split, causal evidence, and full-token targets. Historical output-only
call metrics cannot be compared directly with these full-call costs.

### Request evidence and structured cost scale

`request_surface.py` reconstructs the causally available system, messages
and tools at the request boundary, preserving duplicates and native compaction
semantics. The resulting serialized UTF-8 length is a proxy, with explicit
completeness and provenance fields. It is not an exact provider tokenizer.
The training runner records `feature_schema="causal_request_surface_v2"`;
the comparison adapter selects that feature version from the training report.
Older reports retain their original request-shape features.

The optional `context_scaled` surplus parameterization uses the positive scale
`S = n * output_prior + max(n - 1 - terminal * (n > 1), 0) * (L_in + L_out) / 2`.
The output prior is fitted from positive single-call surplus in training tasks.
The remaining learned ratio allows nonlinear context changes. Inference uses
jointly predicted boundary lengths, and the likelihood includes the triangular
change-of-variables Jacobian. Zero-cost atoms and signed segment algebra are
preserved. `--surplus-parameterization` and `--min-log-length-std` expose the
controlled alternatives; saved configurations determine checkpoint behavior.
`repair_composable_length.py` compares these candidates using the fixed joint
validation likelihood and reports all four positions. Gradient clipping falls
back to a float64 norm only when finite gradients overflow the float32 squared
sum; actual nonfinite gradients still fail.

Unknown execution allowances use a training-only mean feature when configured
with `unknown_allowance_encoding="train_mean"`. This does not impose a call cap.
`adapt_composable_unknown_budget.py` preserves network weights and checks that
known-budget calibration predictions remain identical.

Progress-state coordinate 7 records termination. New pilot training sets
`termination_state_index=7`, ties that coordinate to the sampled terminal flag,
and removes its duplicate likelihood and energy terms. Historical checkpoints
default to no tied coordinate. `adapt_composable_termination_state.py` applies
this support constraint to frozen weights and recalculates source calibration;
its likelihood has a changed contract and cannot establish a training gain.

### Comparing checkpoints

`evaluate_composable_checkpoints.py` verifies identical validation cutoffs and
targets before comparing checkpoints with their original feature schema.
`compare_case_predictors.py --forecasts-only --reuse-baseline-from PATH` verifies
the ordered prompt hashes and reuses frozen baseline responses without API calls.
An optional `--direct-connection` bypasses environment proxies for that gateway
instance only; TLS verification remains enabled. The audit ledger preserves
earlier uncertain attempts, including their unknown monetary charges.

### Request-conditioned method development

`train_request_conditioned_segment.py` evaluates a hierarchical factorization
for call points with a complete, causally visible request:
`p(L | request, model) p(n, stop, L_out, surplus, state_out | L, context)`.
The request prior is a zero-inflated lognormal token/byte ratio fitted only on
training Call Start observations, once per cutoff. Model groups have separate
training statistics; unseen groups use the pooled training distribution.
Inference draws a latent input length and conditions the entire segment sample
on it. The caller's unknown-input flag remains unchanged. Known input lengths
bypass the prior, and an incomplete request surface uses the neural path.

The optional `discretized_lognormal` count head models remaining calls as
`N = 1 + floor(exp(Z))`, where `Z` is Gaussian with two predicted parameters.
For a computational segment of length `h`, a completed segment contributes
`log P(N=n)` and a segment that continues contributes `log P(N>h)`.
At a known execution cap, the final event combines the tail `P(N>=cap)`.
Count and termination are sampled jointly. Count parameters exclude the varying
computational horizon; the continuous cost head still receives segment length.
This optional family preserves the previous categorical checkpoint format.

Optional causal conditions carry completed call count and initial task-text
byte size. Composition increments completed calls by the sampled segment count
and preserves initial task size. These conditions do not depend on execution
caps. The expanded consistency vector includes initial length as well as
`n`, `g`, cost, ending state and termination, so different latent initial
lengths cannot collapse to the same consistency coordinates. Task-hash dropout
uses one mask per training sample, shared by supervised and composed branches.

The optional history-v3 schema exposes the previous completed call's settled
first-request input and remaining charged cost at Task Update. These values
include retry charges and are available at that cutoff. Call Start and In-call
observations retain their existing meaning. Future composed segments clear
these observations, since the next real settled call has not been observed.

The Qwen adapter reads the organized wire requests, complete provider usage and
real observer checkpoints. It rejects truncated or protocol-mismatched runs.
OpenHands checkpoints expose byte counts without prefix text; the adapter
retains this limitation. Its unavailable execution call cap stays unknown.
Task families and repeated executions remain in the same partition across
models and harnesses. New task families use a locked 60/20/20 development/test
split; previous training, validation and calibration memberships remain fixed.

The reproducible expansion uses these modules in order:

```powershell
python -m scripts.experiments.prepare_qwen38_segments --output workspace/runs/qwen38-segments
python -m scripts.experiments.merge_case_training_domains --historical workspace/runs/composable-full-surface-context035 --qwen workspace/runs/qwen38-segments --output workspace/runs/composable-multidomain-data --history-cost-features
python -m scripts.experiments.train_request_conditioned_segment --source workspace/runs/composable-multidomain-data --reference workspace/runs/composable-full-surface-context035-budget-imputed-terminal/model.pt --output workspace/runs/composable-multidomain --call-progress --joint-energy --task-dropout 0.25 --request-prior-groups --task-size
```

The final command fits and selects epochs on development validation NLL.
`--count-distribution discretized_lognormal` enables the experimental count
alternative. Every completed validation saves an atomic epoch weight snapshot
and a progress record. These snapshots preserve weights after interruption; they do
not contain optimizer or RNG state for exact continuation of training.
`--request-input-prior none` with no other architecture flags provides the
previous architecture trained on the same expanded data. Outputs must be empty
to prevent stale selected checkpoints. `evaluate_heldout_case_domains.py`
checks task provenance and freezes model hashes before scoring the separately
stored test set. No module above makes paid API calls. Historical source data
and organized Qwen evidence are local prerequisites and are not bundled in Git.

## Baseline and pairing

`configs/baselines/self_prediction_case_study_v1.txt` preserves the supplied
appendix's zero-shot prompt. Its provenance sidecar records the attachment and
prompt hashes. This is the manuscript's four-position adaptation of
Self-Prediction. The older few-shot prompt remains a separate historical protocol.

Both predictors must receive evidence available at the same actor cutoff, and
both are evaluated against input plus output provider-accounted tokens. The
three online Self-Prediction positions cannot advance the actor or invoke new
tools. Task-start exploration uses a separate copy of the initial environment.
Estimator calls, exploration, elapsed time and token usage are separate overhead.
The baseline's confidence scalar is not a prediction interval.

## Reproduction

Install the repository's `neural` and `dev` extras, then run:

```powershell
python -m pytest tests/test_composable_segment.py tests/test_case_study_evidence.py tests/test_case_study_forecast.py
python -m scripts.experiments.run_composable_case_pilot --runs-root workspace/datasets/swebench-dual-model-v1/recollection-gpt-oauth-v19/runs/runs --output workspace/runs/composable-case-pilot --max-runs 60 --epochs 20
```

Historical captures are local data and are not bundled with the code. The runner
records selected source directories and task-family partitions. It uses the
first complete capture per family in directory order, then a fixed hash ordering
for 60% training, 20% validation and 20% calibration. Repeats cannot cross these
partitions. New case tasks must also be excluded from all three partitions.

The runner makes no network requests. It saves the model, training diagnostics,
calibration predictions and calibrator parameters under the output directory.
Calibration diagnostics are not held-out test results. At 90% nominal coverage,
fewer than nine calibration tasks cannot produce a finite conformal bound under
this rule; the result explicitly retains only an uncalibrated raw interval.
Applying these source calibrators to Claude/DeepSeek with smaller execution caps
is a calibration transfer experiment. It does not establish 90% coverage in that
new setting; paired reports explicitly mark target coverage as unvalidated.

## Execution and accounting

Claude Haiku and DeepSeek Flash are supported by the collection manifests. Gateway-specific
prices require verification; vendor prices cannot establish the user's gateway
bill. Record exact model IDs, reasoning settings, request/output limits, call
counts, actual usage, failures and retries. Unknown monetary charges stay unknown.
No API credential belongs in a manifest, capture, report or Git commit.

Actual paired cases can be evaluated after the implementation and historical
training checks pass. Case selection must be fixed before inspecting relative
prediction errors, and unsuccessful or unfavorable results must remain visible.

The two manifests `case_study_claude_haiku_v1.json` and
`case_study_deepseek_flash_v1.json` each select one low-difficulty authored P0
task and one repeat. Each permits six calls, 1,536 output tokens per attempt,
zero retries and 180 seconds. Its 30,000-token threshold is checked before the
next call; the final call can overshoot it. A budget-stopped capture is retained
only through the evidence reader's explicit diagnostic override. The training
reader still rejects supervision-ineligible runs.

`compare_case_predictors.py` prepares four fixed cutoff prompts per task. Its
baseline ledger allows eight requests, 768 output tokens per request, 60,000
UTF-8 prompt bytes, and a local 150,000-token budget. Byte-based reservations
are conservative local estimates, not gateway billing guarantees. An uncertain
request blocks further calls until its record is reviewed; continuing retains
the full reservation and never resends that request ID. A resumed comparison
must keep its model hash and prompts unchanged.

For subsequent uncapped runs, use the two manifests ending in
`_no_token_call_budget_v1.json` and `compare_case_predictors.py
--no-token-call-budget`. Cumulative logical-call, provider-attempt and provider
token limits are explicitly null. The actor retains an 8,192-token response
limit, zero automatic retries, network timeouts and a 3,600-second wall-clock
guard. Accounting and uncertain-response handling remain active. Existing
bounded captures and ledgers retain their original policies.

A computational prediction horizon is separate from an execution call limit.
When no execution limit is known, `call_limit_known=False` prevents forced
termination at that horizon. Computational horizons do not enter baseline
execution instructions. An unabsorbed sample is marked `horizon_exhausted`;
the comparison refuses to report a full-task estimate from truncated samples.
With `max_rollout_segments > 1`, unresolved unknown-budget particles continue
from their sampled boundary state and context length for additional segments.
Weights remain frozen, known execution caps remain enforced, and first-call
samples remain unchanged. Particles still unresolved at the finite rollout
guard retain the unavailable full-task status. Evaluation reports these failures
explicitly and compares candidates on a common set of available predictions.
`--forecasts-only` evaluates a frozen checkpoint locally without any API call.

`report_case_pilot.py` preserves strict baseline-format failures. It can also
report numerical diagnostics from a response consisting solely of a fenced
JSON object, with the fence removed and numbers unchanged. These values carry
an explicit format-failure marker. Missing network responses are never filled
in. The report separates measured usage from uncertain reservations and writes
a Markdown report plus a LaTeX appendix fragment.

The baseline receives the original seed as a hash-checked static snapshot.
This pilot does not run a separate task-start exploration agent; that protocol
restriction is recorded in every case. Initial dynamic instructions unavailable
at Task Start stay unknown.

## Training diagnostics for task-level errors

`train_request_conditioned_segment.py --weighting family_position` assigns
equal dataset mass to the observed prediction positions, and equal mass to
families within each position. Original segment weights are preserved within
each family/position. This avoids allowing long executions and many repeated
cutoffs to dominate the dataset objective. Mini-batch losses retain their
self-normalized weights. Calibration observations retain their original weights.
The reference checkpoint is rescored with the same validation weights.

`--warm-start --learning-rate 0.0001` fine-tunes the reference weights while
preserving their training-only normalization and request-prior statistics.
Changes to conditioning or coordinate settings are rejected in warm-start mode.
Each completed validation epoch saves an atomic weight snapshot; these snapshots
do not contain optimizer or RNG state and are not exact-resume checkpoints.

Two optional training ablations are available. Setting
`--consistency-first-call-probability 0.5` allocates half of consistency splits
to the first-call/continuation boundary used by inference, while retaining
general random splits. `--energy-coordinates asinh` compares segment samples
in smooth, invertible, logarithmic-tail coordinates; negative context growth
is retained. Its normalization is fitted on training data. The likelihood,
reported token values, and inference samples remain in physical units.
Defaults preserve the previous random splits and physical energy coordinates.
These switches are experimental and do not establish an accuracy improvement.

Use separate output directories for every attempt. Preserve failed runs, select
checkpoints using the specified validation objective, and inspect all four
prediction positions and interval coverage before adopting a candidate. Results
on previously evaluated holdouts provide regression evidence. Independent
confirmation requires fresh test families.
