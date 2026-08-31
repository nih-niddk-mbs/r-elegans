# r-elegans

`r-elegans` is an experimental JAX-native foundation for a differentiable,
connectome-constrained *C. elegans* simulator.

The current milestone contains two independently testable systems:

- a 302-neuron-compatible graded-potential model with masked chemical and
  electrical connectivity, plus Diffrax integration;
- an overdamped planar body model that uses resistive-force theory (RFT) to
  turn prescribed joint-angle waves into locomotion.

It deliberately does **not** claim biological calibration yet. Empirical
connectome ingestion, the neuromuscular adapter, Gymnax environments, and
closed-loop chemotaxis are later milestones described in [PRD.md](PRD.md).

## Development

Use Python 3.10 or newer, create an isolated environment, and install:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Conventions

- Connectivity matrices use `[postsynaptic, presynaptic]` indexing.
- Chemical and gap-junction strengths are nonnegative after transformation.
- Gap-junction masks and strengths must be symmetric.
- Body coordinates use a center-of-geometry frame internally and world-space
  position plus heading externally.

