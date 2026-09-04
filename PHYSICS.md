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
- reduced bending under increased environmental load;
- radius- and length-based Reynolds-number calculations.

## Primary sources

- Fang-Yen et al. (2010), biomechanical measurements and gait adaptation:
  <https://doi.org/10.1073/pnas.1003016107>
- Boyle et al. (2012), Lighthill liquid coefficients, agar calibration, and
  integrated neuromechanics: <https://doi.org/10.3389/fncom.2012.00010>
- Sznitman et al. (2010), swimming at low Reynolds number:
  <https://doi.org/10.1016/j.bpj.2009.11.010>
