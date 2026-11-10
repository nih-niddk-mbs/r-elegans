# Body physics and calibration

## Implemented model

The body is an inextensible planar chain of finite straight rods. Its
generalized velocity contains body translation, body rotation, and all joint
rates. Local resistive-force theory (RFT) gives each rod a drag tensor

```text
D_i = ds (c_parallel t_i t_i^T + c_perpendicular n_i n_i^T).
```

Integrating normal drag over the finite rod also gives rotational drag
`c_perpendicular ds^3 / 12`. The segment Jacobians assemble these terms into a
symmetric generalized resistance matrix `R`. A Schur complement eliminates
rigid translation and rotation under zero net force and torque. The remaining
joint system balances environmental resistance, passive Kelvin--Voigt bending,
and active muscle moments. Elasticity is stepped implicitly for stability.

Physics matrix products explicitly request `jax.lax.Precision.HIGHEST`. On
accelerators such as the A100, float32 arrays can otherwise use reduced-mantissa
matrix multiplication. Cancellation in resistance assembly and the Schur
complement makes that shortcut inappropriate here. This preserves the existing
physics-test tolerances rather than relaxing them for GPU results. A separate
float64-reference test checks resistance accuracy on the executing backend.

This coupling is important: absolute environmental drag now changes the
response to a muscle command. The old implementation prescribed joint rates
before applying drag, so viscosity could not resist bending.

## Physical modes

`physical_muscle_body_params(..., medium="liquid")` uses:

- adult length `1.0 mm` and radius `40 um`;
- water/M9 viscosity `1.0 mPa s` by default;
- Lighthill coefficients with `q = 0.09 wavelength` and default wavelength
  `1.5 mm`;
- measured bending modulus `9.5e-14 N m^2`;
- measured internal-viscosity upper bound `5e-16 N m^2 s`;
- muscle activation time constant `100 ms`.

`physical_muscle_body_params(..., medium="agar")` uses effective whole-worm
drag `C_parallel = 3.2e-3 kg/s` and `C_perpendicular = 128e-3 kg/s`, giving a
ratio of 40. The values are divided by body length before the local RFT solve.

The maximum active moment is parameterized as a preferred curvature of
`10 mm^-1` acting through the measured passive bending modulus. This is a
documented modeling closure, not a direct measurement of muscle force.

## What “correct” means here

This is a physically dimensioned, dissipative reduced-order model appropriate
for fast differentiable simulation and control. It is not full computational
fluid dynamics. In particular it does not yet include non-local hydrodynamic
interactions, walls, free surfaces, self-contact, agar deformation, adhesion,
or a resolved lubrication film. Agar coefficients are therefore an effective
calibration rather than a microscopic surface model.

The legacy `default_muscle_body_params` remains normalized so existing fitted
controllers can be inspected and reproduced. A normalized dish and an SI body
must never be combined. WormGym environments will select their body and world
unit system together, and physical gaits will be fitted before RL begins.

## Validation

`tests/test_physical_calibration.py` checks:

- the expected liquid drag anisotropy near 1.5;
- the published agar coefficients and ratio 40;
- conversion of continuum bending properties to discrete joints;
- symmetry and positive definiteness of the resistance matrix;
- float32 resistance accuracy against a float64 reference;
- reduced bending under increased environmental load;
- radius- and length-based Reynolds-number calculations.

## Primary sources

- Fang-Yen et al. (2010), biomechanical measurements and gait adaptation:
  <https://doi.org/10.1073/pnas.1003016107>
- Boyle et al. (2012), Lighthill liquid coefficients, agar calibration, and
  integrated neuromechanics: <https://doi.org/10.3389/fncom.2012.00010>
- Sznitman et al. (2010), swimming at low Reynolds number:
  <https://doi.org/10.1016/j.bpj.2009.11.010>

## Medium-physics audit status (E045, 2026-11-10)

Current trained winner uses the12-rod SI liquid factory:1mm length,40um
constant radius,water-like viscosity0.001Pa s,Lighthill local RFT. Code/source
comparison finds the expected per-length coefficients0.00329041 parallel and
0.00521525 perpendicular kg/(m s),ratio1.58498. Whole-body coefficients are
those values times0.001m. Coupled torque-driven mechanics impose zero net
force/torque and include external drag in shape dynamics.8 targeted existing
physics tests pass (units,SPD/precision,residual balance,load response,Re).
These are consistency checks,not quantitative experimental calibration.

Status:physically dimensioned leading-order UNBOUNDED NEWTONIAN approximation.
No explicit Petri-dish bottom/wall/free-surface flow,nonlocal hydrodynamic
interactions,body taper or self-contact. Agar mode is a different effective
surface-drag closure,not resolved gel/adhesion/lubrication. Current liquid
winner does not establish physical crawling or a real Petri-dish environment.
Water retains ordinary viscosity;microscale viscous dominance is a scale effect.

Fixed drag wavelength1.5mm is a reference parameter and is not measured from
the learned standing-like shape. Passive viscosity uses a reported upper bound;
active muscle moment scale is a preferred-curvature modeling closure. Neither
is a measured complete actuator/medium calibration. No matched observed
waveform→predicted force/speed benchmark or12→24→48 spatial convergence has
been established. Any such refinement must keep physical curvature and muscle
interpolation consistent,not just increase the segment count. Current neural
branch lacks body-to-brain feedback,so changing drag does not adapt its neural
rhythm through proprioception. That is separate from the force solver.

For scale only,finite differences of copied true-world preview nodes estimate
maximum sampled speed0.194mm/s,Re~0.00777(radius)or0.194(length);these are not
full-resolution fluid velocities or a hydrodynamic-error bound. They support
a viscous-dominated local approximation without certifying all inertial/flow
effects. No new integration or trained parameter changes.

Primary-source context:Boyle etal(2012) uses the same Lighthill convention
(https://pmc.ncbi.nlm.nih.gov/articles/PMC3296079/). Fang-Yen etal(2010) measured
~9.7x transverse drag in a150um glass chamber compared with bulk fluid
(https://www.wenlab.org/wordpress/3.8/wp-content/uploads/2013/12/proc_natl_acad_sci_usa_2010_fang-yen.pdf).
That factor is geometry-specific;do not blindly multiply bulk drag or viscosity.

Recommendation:before quantitative locomotion claims or further medium-specific
training,benchmark body/medium response to a prescribed measured waveform,
check resolution/time convergence and choose bulk versus wall/surface geometry.
Keep7001 as an amplitude primitive under its existing model. Audit record:
`experiments/records/E045-medium-physics-audit.json`.

## Wet-surface reference assessment (E046)

The primary natural-behavior target is now movement on a moist, compliant
surface. Standard **1.7% wet NGM agar** is the first reproducible laboratory
proxy; it is not a claim to reproduce the material properties of decaying fruit.
`body/wet_surface.py` implements an independent float64 reference solver using
the measured force-speed fits of [Rabets et al. (2014)](https://doi.org/10.1016/j.bpj.2014.09.006).
For a 1mm contact length, with speed in micrometers/s and force in microNewtons:

```text
F_t = 0.32 |v_t|^0.58
F_n = 6.8  |v_n|^0.31
```

These are published fitted curves, not raw observations. Convert each to force
per length before quadrature; do not assign whole-worm force to every rod. The
physical force opposes velocity. Other supported, separately measured agar
concentrations are 1.0%, 3.0%, 4.0% and 6.0%; no concentration interpolation is
implied. The previous constant-drag `medium="agar"` is retained as a legacy
comparison and is not silently changed into this nonlinear law.

At zero velocity the power-law slope diverges. The reference uses smooth odd
resistance `a*r*(r*r+eps*eps)^((gamma-1)/2)`, where `r=v/(1um/s)` and `eps` is
the regularization speed in the same units. Default smoothing is **0.1um/s**;
0.01 and 1um/s are sensitivity cases, not measured parameters. Very weak
motion depends on this unmeasured extrapolation. This model is dissipative
with a positive Hessian; Newton solves enforce free rigid force/torque and
joint moment balance, with backward-Euler passive elasticity. Failed solves
raise errors. Finite rods use four-point Gauss quadrature, with eight-point
comparison. Quadrature refinement is not body-segment spatial convergence.

The assay generates the pinned seed7001's endogenous 95-muscle output once,
then replays the identical commands into each body model, starting straight
and relaxed. This preserves the current **one-way** brain-to-body dynamics
because there is no body-to-brain feedback. No sinusoidal teacher, external
current or parameter fitting is introduced. Holding commands across their
recorded intervals makes the half-timestep replay a body/muscle integration
check; it does not refine the neural integration. A held-voltage control uses
the actual static NMJ output, not artificially zero muscle activations.

Run on mbslnx2 after pulling:

```sh
cd /home/carsonc/github/r-elegans
git pull --ff-only
sh scripts/assess_wet_surface_mbslnx2.sh
```

The launcher pins the successful external checkpoint and frozen readout by
SHA256, uses the absolute dedicated environment's Python and four physical
CPU cores, and detaches with nohup. Defaults:60 simulated seconds per case,
20-minute neural-generation budget; body replays follow. The log path and PID
are printed. `WET_SURFACE_RUN_EXIT_CODE=0` means the assessment completed, not
that crawling was learned. `summary.json`, `force_reference.png`, per-case
`trajectory.png` and true-geometry `motion.html` remain outside Git under
`/home/carsonc/r-elegans-results/wet-surface-*`. Commands, all recorded neural
states, motor outputs, body trajectories, input hashes, source and environment
are retained. No optimizer or random seed is used in this deterministic assay.

This reference is not yet the JAX training environment. It does not resolve
capillary/adhesion forces separately, groove memory, full contact, self-contact
or natural substrate microstructure. Towing-fit reproduction checks units and
the effective law; independent gait/force validation and appropriate low-speed
data are still needed before quantitative biological crawling claims.

## Sensory feedback and the wet reference (E048)

The new opt-in embodied sensory branch computes actual body curvature and head
position before each neural interval, then updates muscles and body. This is
implemented for the JAX liquid body and for the NumPy nonlinear wet reference.
`surface_body_step` matches the existing reference mechanics and adds no new
force scale. The wet body remains a slow reference with the E046 uncertainty
in its low-speed empirical force closure.

E046 command replay is still valid for its historical one-way/no-input assay.
It is not valid for E048: with body-to-neuron feedback, each medium requires its
own interleaved neural/body trajectory. The new diagnostic performs that
integration rather than reusing a liquid neural recording for wet crawling.
Chemical test fields are prescribed local-concentration fixtures, separate from
fluid mechanics and unvalidated as natural microbial plumes. No contact/water
sensor, chemical fluid transport or proprioceptive molecular mechanism is
claimed by adding this causal connection.
