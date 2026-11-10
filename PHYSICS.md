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
