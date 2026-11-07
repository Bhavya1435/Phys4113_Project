# PHY4113 numerical simulations: Tasks 1 and 2

These scripts solve the full nonrelativistic Lorentz equations and compare
measurements with the leading guiding-center predictions in the professor's
*Charged Particles in Magnetic Fields — Adiabatic Invariants and Guiding-Center
Theory* project description, Tasks 1 and 2, equations (1)–(4). The course PDF is
read locally and is not redistributed. Tasks 3–5 are not implemented; Task 6
and the project's theoretical derivations remain the partner's responsibility.

OpenAI Codex was used to assist in writing, debugging, and validating the
numerical code. Numerical values below were obtained by executing the code;
the benchmark numbers are used only for comparison.

## Run and reproduce

From the repository root, with Python 3 and NumPy, SciPy, and Matplotlib:

```powershell
python -m pip install -r requirements.txt
python Simulations/01_particle_drifts.py
python Simulations/02_magnetic_bottle.py
python Simulations/validate.py
```

Each task reruns all three tolerance settings, overwrites its own generated
figures and results, and checks physical and convergence thresholds. Execution
is headless; no display, network connection, course PDF, or external data is
needed after installing dependencies. Output paths are relative to the scripts,
so execution also works from another working directory. Each task takes roughly
one to a few minutes depending on the machine. Version ranges in the root
requirements support installation; the exact versions used for the checked-in
outputs are recorded in each summary JSON (Python 3.14.0, NumPy 1.26.4,
SciPy 1.16.3, Matplotlib 3.10.7). Other compatible versions may change last digits.

All axes use consistent normalized units with q=m=B0=1: the reference
gyrofrequency is 1 and reference gyroperiod is 2π. No physical SI scale is
specified by the project. Velocities, lengths, time, energy, and magnetic moment
are consequently reported in these normalized units.

## Shared equations and methods

The six-component state is `(x,y,z,vx,vy,vz)`. `common.integrate` solves

\[
\dot{\mathbf x}=\mathbf v,\qquad
\dot{\mathbf v}=\frac qm(\mathbf E+\mathbf v\times\mathbf B)
\]

with `solve_ivp(method="DOP853", dense_output=True)`. Runs use
`(rtol,atol)=(1e-8,1e-10),(1e-10,1e-12),(1e-12,1e-14)`.
Uniform output grids are evaluated from the solver's dense interpolant and do
not determine the adaptive integration steps. The tightest run produces figures
and trajectory CSVs. There is no Boris or guiding-center orbit integrator here.

Diagnostics use the field at the **particle position**:

\[
B=|\mathbf B|,\quad \mathbf b=\mathbf B/B,\quad
u=\mathbf v\cdot\mathbf b,\quad
\mathbf v_\perp=\mathbf v-u\mathbf b,\quad
\mu_{inst}=\frac{m|\mathbf v_\perp|^2}{2B},\quad
K=\frac12m|\mathbf v|^2.
\]

The local first-order estimate of the guiding center is

\[
\mathbf X=\mathbf x+\frac{m\mathbf v\times\mathbf B}{qB^2}.
\]

For the uniform E case, replace v in this map by v−v_E to remove the drift
from the gyroradius. For a nonuniform field this map has residual higher-order
and gyrophase dependence; it is not an exact transformed coordinate. The shared
integrator accepts field callables, arbitrary q and m, and solver events. These
interfaces support future mirror fields, off-axis comparisons, and electrostatic
trap fields without implementing any of those tasks now.

## Task 1: particle drifts

Both cases use q=m=B0=1, initial laboratory perpendicular speed 0.5, and duration
`200*2π = 1256.6370614`, sampled at 40,001 times.

**Uniform E and B:** B=(0,0,1), E=(0.05,0,0). Initial state is
`(0,0,0,0.5,0,0)`. The initial velocity points along +x. The gyration speed in
the drift frame is sqrt(0.5²+0.05²); the specified laboratory speed is 0.5.
The theoretical drift is

\[
\mathbf v_E=\frac{\mathbf E\times\mathbf B}{B^2}=(0,-0.05,0).
\]

For the chosen initial state the exact orbit is
`x=0.5 sin(t)+0.05[1−cos(t)]`,
`y=-0.05t+0.5[cos(t)-1]+0.05 sin(t)`, `z=0`, with velocities obtained by
differentiation. Kinetic energy alone is not conserved: the conserved total
energy is `H=K−q E0 x`, since the potential is `φ=−E0 x`.

**Gradient B:** B=(0,0,1+x/100), E=0, initial state
`(0,0,0,0.5,0,0)`. This puts the initial local guiding center at x=0.
The reference μ=m v_perp²/(2 B0)=0.125, so

\[
\mathbf v_{\nabla B}=\frac{\mu}{qB_0}\mathbf b\times\nabla B
=(0,0.00125,0).
\]

This drift-test field is divergence-free but is not a vacuum (curl-free)
field. K is conserved, and B remains positive throughout the computed orbit.

The measured drift is the slope of an ordinary least-squares linear fit to
`Y(t)` over the full duration, with a free intercept. Absolute theory error is
`abs(v_fit−v_theory)` and relative error divides by `abs(v_theory)`. Eight
nonoverlapping time blocks provide a fit-window sensitivity diagnostic: the
reported range of block slopes is not a statistical confidence interval.
Deterministic gyrophase residuals invalidate interpreting pointwise OLS errors
as independent random noise.

| Quantity | Theory | Tightest simulation | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| E×B drift, y component | −0.05 | −0.05000000000000024 | 2.36e−16 | 4.72e−15 |
| Gradient-B drift, y component | 0.001250 | 0.001250032783318355 | 3.28e−8 | 2.62e−5 |

The professor quotes drift **magnitudes** 0.05 and 0.001250; signs here follow
the documented field orientations. The PDF's sample E×B measurement 0.05002 is
not an exact target. In a uniform field, the local guiding-center Y obeys
`dY/dt=−E0/B0` algebraically, so its slope is exceptionally insensitive to
integration errors. Roundoff-level agreement in the drift is **not** evidence
of a roundoff-accurate full orbit. The independent exact-orbit comparison gives
a maximum componentwise state error of 6.73e−11.

For gradient B, the 3.28e−8 drift difference persists after tightening the
solver; the last tolerance change is only 1.63e−12. The block-slope range is
1.55e−10. These diagnostics indicate a finite-gyroradius/theory difference
rather than integration error. Leading theory evaluates B and μ at the nominal
center; the finite orbit and local center estimate retain higher-order effects.

## Task 2: magnetic bottle

The prescribed cylindrical vacuum field is implemented in Cartesian coordinates
without a singularity on the axis:

\[
B_x=-B_0xz/L^2,\quad B_y=-B_0yz/L^2,\quad
B_z=B_0[1+(z^2-(x^2+y^2)/2)/L^2].
\]

Parameters are q=m=B0=1, L=10, v_perp=0.5, v_parallel=0.2, and t=0…700.
Initial z=0 and gyrovelocity points along −y. We place the initial **local
guiding center** exactly on the magnetic axis by solving

\[
\rho[1-\rho^2/(2L^2)]=0.5,
\]

then setting `(x,y,z)=(ρ,0,0)` and `(vx,vy,vz)=(0,−0.5,0.2)`.
The small positive root is ρ=0.500627355536308. At z=0 the field is vertical,
so the specified local parallel and perpendicular speeds are exact. This is a
first-order center convention, not an assertion of an exact all-orders center.
Setting the particle itself at x=y=0 with nonzero v_perp would put its guiding
center off-axis. Using ρ=0.5 instead changes the initial local center slightly.
The PDF leaves this finite-radius mapping unspecified.

On axis the leading guiding-center energy is
`H_GC=m u²/2+μbar B0(1+Z²/L²)`. With
`μbar=m v_perp²/(2 B0)`, the predicted axial oscillator is

\[
\omega_b=v_\perp/L=0.05,\quad
Z(t)=\frac{Lv_\parallel}{v_\perp}\sin(\omega_b t),\quad
T_b/2=\pi L/v_\perp,\quad Z_{max}=Lv_\parallel/v_\perp.
\]

This prediction uses B at the **guiding center**; the instantaneous initial
particle μ is 0.125156838884077 rather than 0.125 because the finite-radius
particle starts in a slightly weaker field. Substituting that instantaneous μ
into the nominal theoretical formula would change the comparison convention.

**Measurement definitions:** z(t) is the particle's Cartesian axial position.
`half_period` is the mean interval between successive z=0 crossings after
discarding the initial crossing; opposite crossing directions count as a
half-period. `full_period` is independently averaged over every other crossing.
Events are root-located using the dense solver interpolation. `zmax` is the
largest absolute z among all vz=0 turning points over 0<t≤700, rather than a
sampled maximum or an average amplitude. Eleven turning points and eleven
post-initial midplane crossings are recorded in JSON.

| Quantity | Guiding-center prediction | Tightest full orbit | Professor's benchmark |
|---|---:|---:|---:|
| Axial half-period | 62.8318530718 | 62.8009850510 | ≈62.80 |
| Axial full period | 125.6637061436 | 125.6019704746 | — |
| Maximum absolute z | 4 | 3.9976658819 | ≈3.998 |
| Relative peak-to-peak μ variation | 0 at leading order | 2.9332370e−4 | order 3e−4 |

The half-period differs from leading theory by 0.0308680 (0.04913%), and the
maximum displacement by 0.00233412 (0.05835%). Individual half-periods range
from 62.80098251 to 62.80098802; turning amplitudes span 6.08e−6. Small finite
orbit effects make the full-orbit bounce slightly different from a perfectly
harmonic guiding-center oscillator. The differences survive tolerance
refinement, and the finite-radius initial field and approximate μ mapping also
affect comparison with the leading on-axis model. These results reproduce the
benchmark to its quoted precision; they do not establish a unique higher-order
decomposition of the discrepancy.

**Magnetic moment:** use the magnitude and direction of the full local field,
not Bz alone or vx²+vy² alone. On the saved output grid:

- Absolute peak-to-peak variation `max(μ)−min(μ)` = 3.6705772e−5.
- Relative peak-to-peak variation `[max(μ)−min(μ)]/mean(μ)` = 2.9332370e−4.
- Maximum relative departure from the initial value `max(abs(μ/μ(0)−1))`
  = 2.9055176e−4.

The PDF's phrase “constant to 3e−4” does not specify a normalization. Both
relative definitions agree with that scale; the absolute variation is a
different number. Instantaneous μ is only an approximate invariant, so its
resolved oscillation is physical and need not vanish as tolerances tighten.
The initial η≈ρ/L≈0.05 describes the expansion scale; the on-axis symmetry
suppresses the radial first-order gradient correction, allowing much smaller
variation here. μbar is not calculated by a higher-order Lie transform in this
implementation.

## Numerical verification and limitations

| Diagnostic | rtol=1e−8 | rtol=1e−10 | rtol=1e−12 |
|---|---:|---:|---:|
| Max E×B total-energy error / initial K | 6.93e−7 | 4.17e−9 | 2.70e−11 |
| Max gradient-B relative K error | 5.73e−7 | 3.44e−9 | 2.23e−11 |
| Max bottle relative K error | 3.17e−7 | 1.92e−9 | 1.26e−11 |
| Bottle half-period | 62.8009904029 | 62.8009850812 | 62.8009850510 |
| Bottle max absolute z | 3.9976660137 | 3.9976658827 | 3.9976658819 |

Last-two-tolerance changes in the bottle are 3.01e−8 in half-period, 7.59e−10
in maximum z, 8.46e−10 in relative μ range, and 6.13e−8 in the largest sampled
componentwise state difference over the full orbit. These are empirical
convergence estimates, not rigorous bounds. Full-orbit state errors can
accumulate as phase error even when energies are nearly conserved. DOP853 is
not a geometric long-time integrator; validation here applies to the stated
durations.

Output spacing is about 0.0314 for Task 1 and 0.02 for Task 2, resolving the
gyroperiod. Task 2 additionally reevaluates μ on a 0.01 grid to check extrema
sampling; the refined range and its difference are recorded under `sampling`
in its JSON. Bounce event locations are independent of these output grids.
The refined relative μ range is 2.9332378056e−4, a change of only 7.86e−11.

`validate.py` independently checks the exact uniform-B cyclotron orbit with
both charge signs, finite-difference divergence and curl of the bottle field,
saved drift directions and error thresholds, the initial on-axis center, and
all expected PDF/PNG file signatures. The three tests passed for the saved run.
The E×B exact trajectory check is also calculated during Task 1. Scientific
assertions in both task scripts must pass before successful exit. The small
smooth changes in μ are many orders larger than the energy error and persist
under tolerance refinement, separating them from numerical energy drift.

## Outputs and figure captions

All seven figure stems below produce **vector PDF and 350-DPI PNG**, with
embedded TrueType fonts in PDF. Plot text is sized for enlargement on a poster.
Captions below can be adapted to the poster; they are not embedded as dense
text inside the figures.

| Figure stem in `figures/` | Scientific caption |
|---|---|
| `task1_orbits_and_drifts` | Full particle orbits and local guiding centers for uniform crossed fields and a linear magnetic gradient. Left panels show the first eight reference gyroperiods with equal coordinate scales; right panels compare fitted drift and leading theory over 200 periods. Field orientations determine the opposite drift signs. |
| `task1_fit_residuals` | Residual guiding-center displacement during the first eight gyroperiods after removal of the linear y drift fitted over the complete run. The gradient case retains a small deterministic oscillation because the local guiding-center map is approximate; the uniform case is limited by roundoff. |
| `task1_convergence` | Changes in fitted drift relative to the tightest run, and maximum conserved-energy errors, as solver tolerances tighten. Uniform-field drift extraction has algebraic cancellation and reaches roundoff before the full orbit does. |
| `task2_bounce` | Axial motion in the vacuum bottle compared with the leading harmonic guiding-center prediction. The lower panel resolves the accumulating phase and amplitude difference over approximately 5.6 bounce periods. |
| `task2_invariants` | Relative departure of instantaneous local-field magnetic moment from its initial value, scaled by 10⁴, and relative kinetic-energy error. Resolved adiabatic variation is about 3e−4 while numerical energy error remains about 1e−11. |
| `task2_orbit` | Three-dimensional full particle orbit in the bottle, showing fast gyration and slow axial bounce. The plotted curve is subsampled by five for legibility; integration and diagnostics use the complete time grid. Axis ranges differ and should be read numerically. |
| `task2_convergence` | Half-period and turning-amplitude changes relative to the tightest run demonstrate convergence independently of the discrepancy from leading guiding-center theory. |

`results/task1_summary.json` and `results/task2_summary.json` contain parameters,
initial conditions, measured quantities, tolerance comparisons, event records,
and package versions. CSV files `task1_exb.csv`, `task1_gradb.csv`, and
`task2_orbit.csv` contain the tightest trajectories with named columns.
Task 1 also saves local guiding-center coordinates; Task 2 saves μ and K.
No reference value is substituted for a computed result.

Existing root README, `Core Theory`, `References/`, and `Simulations/Placeholder`
are preserved. No course documents or additional textbook files are included.
