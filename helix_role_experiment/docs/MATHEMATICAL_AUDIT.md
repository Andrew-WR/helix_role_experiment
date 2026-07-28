# Mathematical audit of the original extraction

## Scope and provenance

The only original artifact present in this checkout is the extraction and
router calibration code in `../router.py`. No paper, PDF, notebook, training
code, fine-tuned checkpoint identifier, or original activation dataset was
supplied. Consequently this audit distinguishes:

- claims proved from the extraction mathematics;
- implementation facts verified in `router.py`; and
- empirical questions that remain untested until the missing artifacts or new
  traces are available.

The PLL and all downstream routing logic are out of scope.

## Exact estimand

The target estimand is not the phase of a per-output `k=1` reconstruction.
That phase is largely specified by the analysis. The target is:

> The held-out, problem-level causal effect of manipulating a frozen shared
> activation subspace on semantic computational-state transitions, conditional
> on sequence position, procedural operation, confidence, and termination
> affordance, together with the unique observational variation in that
> subspace explained by each of those candidate variables.

The candidate plane is called the **candidate low-frequency subspace** until it
passes every semantic-progress criterion in the pre-registration.

## What the per-trace Fourier analysis guarantees

For a centered trace \(X\in\mathbb{R}^{L\times d}\), the real `k=1`
reconstruction is

\[
\hat x_t=a\cos(2\pi t/L)+b\sin(2\pi t/L).
\]

It therefore:

1. has rank at most two;
2. lies in `span(a,b)`;
3. completes one revolution over the observed trace;
4. has imposed angular frequency \(2\pi/L\); and
5. exists for any nonconstant trace with nonzero first-harmonic energy.

Let \(M=[a\ b]\) and \(q_t=(\cos\omega t,\sin\omega t)^T\). Then
\(\hat x_t=Mq_t\), and over a complete sampled revolution its covariance is
approximately \(\frac12MM^T\). If \(M=UDR^T\), exact two-dimensional PCA
whitening gives

\[
D^{-1}U^T\hat x_t=R^Tq_t.
\]

Thus `atan2` after correct per-trace whitening recovers normalized output
position up to orientation and offset. Figure 1 and the tautology table are
diagnostics of this construction, not evidence of a model representation.

## Verified implementation issues in `router.py`

### One revolution is imposed

`isolate_k1_and_residual` centers each completed trace, retains only `rfft`
bin 1, and applies `irfft`. Any resulting ellipse or circle is mathematically
expected.

### Gauge-unsafe frame averaging

`build_helix_basis_cache` fits PCA independently to every reconstructed trace.
It aligns the two axes with independent sign flips and then averages PC1 and
PC2 separately. This does not resolve axis swaps, reflections, or arbitrary
rotations when the ellipse is close to circular. A stable plane can coexist
with unstable individual axes.

### Residual-direction removal is unjustified

The code estimates PCA directions from the non-`k=1` temporal residual and
projects those feature-space directions out of the averaged basis. Temporal
Fourier orthogonality does not imply activation-space directional
orthogonality. This can remove genuine modulation, stalls, reversals, or
sidebands.

### Future information enters calibration

Each trajectory is centered by its complete-output mean and its Fourier basis
uses final length \(L\). Those quantities are unavailable online. They are
valid for retrospective spectral analysis but cannot define an online
progress variable without a separate causal estimator.

### Exceptions are silently swallowed

The calibration loop catches every exception and continues. This can select a
nonrepresentative subset without an auditable exclusion record.

### Activation sampling needs explicit token alignment

The hook records only calls whose sequence dimension equals one. Depending on
the generation implementation, this can omit prefill and create a one-token
offset between the activation producing a token and the recorded token. The
new collector stores prompt length, generated token ID, activation token
index, and hook call index explicitly.

## Required empirical audit

The package reports:

- exact whitened phase recovery versus \(2\pi t/L\);
- first-harmonic concentration against shuffle, phase-randomized, random-walk,
  drift, polynomial, autoregressive, and boundary nulls;
- sensitivity to detrending, windowing, DCT-style nonperiodic bases, reflection,
  and endpoint treatment;
- endpoint-distance and fitted-trend contributions;
- per-trace eigengaps and radius reliability;
- principal angles, chordal distances, and projector similarities;
- bootstrap intervals over problems, never pooled token pseudo-replication;
- held-out spectral selectivity for three gauge-correct estimators.

## Conclusions allowed before new data

The available mathematics establishes that the original per-output phase is
predominantly normalized position by construction. It does **not** establish
that raw activations contain no shared low-frequency plane, nor that such a
plane lacks a causal role. Those are empirical questions addressed by frozen,
held-out shared-basis analysis and crossed causal interventions.

