# PHY4113 numerical simulations: Tasks 1, 2, 3, and 5

These scripts solve the full nonrelativistic Lorentz equations and compare
measurements with the leading guiding-center predictions in the professor's
*Charged Particles in Magnetic Fields — Adiabatic Invariants and Guiding-Center
Theory* project description. Tasks 1–3 use equations (1)–(4), and Task 5 also
uses equation (5). The course PDF is
read locally and is not redistributed. Task 4 is not implemented; Task 6
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
python Simulations/03_loss_cone.py
python Simulations/05_penning_trap_bhavya.py
python Simulations/validate_tasks_3_5.py
```

Each task reruns all three tolerance settings, overwrites its own generated
figures and results, and checks physical and convergence thresholds. Execution
is headless; no display, network connection, course PDF, or external data is
needed after installing dependencies. Output paths are relative to the scripts,
so execution also works from another working directory. Tasks 1 and 2 take roughly
one to a few minutes depending on the machine; Task 3 takes substantially longer
because of its many scans and extended-time checks. Version ranges in the root
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

## Task 3: periodic magnetic mirror and loss cone

`03_loss_cone.py` implements the PDF's periodic **vacuum** field using
`scipy.special.i0` and `i1`. Put r=sqrt(x²+y²), s=kr. The Cartesian form is

\[
B_x=B_0\delta k\frac{I_1(s)}s\sin(kz)x,\quad
B_y=B_0\delta k\frac{I_1(s)}s\sin(kz)y,\quad
B_z=B_0[1+\delta I_0(s)\cos(kz)].
\]

This follows directly from (Bx,By)=Br(x,y)/r. The ratio I1(s)/s tends to 1/2,
so the transverse components vanish regularly on axis. For |s|<1e−4 the code
uses `1/2+s²/16+s⁴/384`, avoiding a division by zero. The Bessel identity
`d[r I1(kr)]/dr=kr I0(kr)` gives div B=0; equality of ∂Br/∂z and ∂Bz/∂r
gives curl B=0. Independent finite differences verify both properties,
including the axis limit.

Parameters are q=m=B0=1, δ=0.5, k=0.05, and the prescribed main observation
duration is 900. The PDF does **not** prescribe a total speed for Task 3.
This implementation chooses v=0.5, holds it fixed throughout the angle scan,
and records it in JSON. Finite-time excursions and crossing times depend on
this choice; the leading loss-cone angle does not.

The field minimum is z0=π/k=62.8318530718. Adjacent maxima occur at z=0 and
2π/k=125.663706144. In the primary convention the **physical particle** starts
at `(0,0,z0)` with `(vx,vy,vz)=v(sin α,0,cos α)`. This follows the Task 3 wording
“particle starting on the axis at the field minimum.” Both the speed and the
pitch relative to the local B are exact. Its first-order local guiding center
is displaced to `(0,−v sin α/[B0(1−δ)],z0)`. It is therefore different from
putting the guiding center on axis, as requested for Task 2.

Initialization sensitivity is also checked using the alternative convention
`x=ρ, y=0, z=z0, vx=0, vy=−v sin α, vz=v cos α`, where the small positive root
of `ρ B0[1−δ I0(kρ)]=m v sin α/q` makes the local first-order center lie on
axis. These are distinct physical launches at the same local pitch angle;
neither is an exact all-orders transformed guiding-center initialization.

With approximate conservation of μ and exact conservation of energy,
reflection requires Bturn=Bmin/sin²α to lie below Bmax. Thus

\[
R=\frac{1+\delta}{1-\delta}=3,\qquad
\sin^2\alpha>\frac1R,\qquad
\alpha_c=\arcsin(1/\sqrt3)=35.2643896828^\circ.
\]

Equality is the guiding-center separatrix. Near it the transit/bounce time
becomes large, so lack of a crossing during a fixed window is not sufficient
evidence of confinement. The strict classification used in the code is:

- **escaped:** a solver event records crossing either adjacent maximum plane,
  `abs(z−z0)=π/k`. This means leaving the initial mirror cell, not leaving the
  infinitely periodic magnetic system.
- **bouncing_observed:** there is no barrier crossing in the observation
  window and axial vz=0 turning points have occurred on **both sides** of z0.
  This provides positive evidence of repeated bounce motion over that window.
- **unresolved:** neither of those conditions is met. A non-crossing particle
  with no reversal, or with only a first reflection, is not labeled trapped.

`max_axial_excursion` means max|z−z0| over the stated duration. It is calculated
from solver-located vz=0 extrema and both endpoints, rather than just plot
samples. Escaping orbits continue to t=900, so their maximum excursions may
span multiple cells. The on-axis guiding-center turning prediction, when it
exists, is

\[
\Delta z_{turn}=\frac1k\arccos\left[
\frac{1-(1-\delta)/\sin^2\alpha}{\delta}\right].
\]

The coarse scan includes 25, 28, 31, 34, 35, 36, 39, 42, 45, 50, 55, and
60 degrees. A 0.01-degree grid covers 35.20…35.32 degrees, and the exact
analytical threshold is also simulated. These main scans use rtol=1e−10,
atol=1e−12. The observed 900-time bracket is refined until an unresolved
midpoint is reached; unresolved outcomes are retained in the output.
A **separate extended-time bracket** uses t=3600 and rtol=1e−12, atol=1e−14
with a target width of 1e−4 degrees. It also stops if a midpoint remains
unresolved. Its two endpoints are repeated at 1e−10/1e−12 to check numerical
sensitivity. This bracket is specific to the speed, launch convention, and
finite duration; it is not a rigorous infinite-time loss-cone boundary.

Reference angles and both sides of the 900-time bracket use all three
tolerance pairs from Tasks 1–2. Separate tight runs to t=1800 are inspected
at nested horizons 225, 450, 900, and 1800, separating delayed decisions from
numerical error. The main scans remain t=900 throughout. Run times are longer
than Tasks 1–2 because many orbits are integrated; progress is printed per case.

Measured main-window results (tight reference runs where available; other
angles are the 1e−10 scan):

| Pitch (degrees) | Main-window classification | Maximum excursion | First barrier time | On-axis GC turning prediction |
|---|---|---:|---:|---:|
| 25 | escaped | 352.5668831 | 159.2282597 | no first-cell turning point |
| 35 | escaped | 181.3113125 | 337.2125484 | no first-cell turning point |
| 35.22 | escaped | 73.6030888 | 547.8978771 | no first-cell turning point |
| 35.23 | unresolved: one reflection | 61.9207812 | none by 900 | no first-cell turning point |
| 35.26438968 | unresolved: one reflection | 60.6172343 | none by 900 | limiting separatrix, not a finite bounce |
| 36 | bouncing observed: two reversals | 53.3006179 | none by 900 | 53.5589009 |
| 45 | bouncing observed | 31.3487297 | none by 900 | 31.4159265 |
| 60 | bouncing observed | 16.7828358 | none by 900 | 16.8213734 |

Thus the specified 35-degree/36-degree benchmark is reproduced. The two
relative 36-degree excursion predictions are particularly informative:
the primary particle-axis orbit differs from on-axis guiding-center theory
by about 0.482%, while the alternative local-center-axis orbit reaches
53.5433512, about 0.0290% below that theory. Changing the finite-radius
initialization changes the physical orbit appreciably near the loss cone;
it is not an integration-tolerance adjustment or a substituted benchmark.

All 27 stored main-window angles conserve K to within 2.97e−9 relative
at rtol=1e−10. The tight reference/boundary runs conserve K to within
1.92e−11. The convergence details are:

| Pitch | Relative K error, 1e−8 | Relative K error, 1e−10 | Relative K error, 1e−12 | Last-two excursion change | Last-two barrier-time change |
|---|---:|---:|---:|---:|---:|
| 35 | 3.71e−7 | 2.30e−9 | 1.55e−11 | 1.70e−7 | 9.26e−7 |
| 35.22 | 4.81e−7 | 2.97e−9 | 1.91e−11 | 1.60e−5 | 7.39e−5 |
| 36 | 3.77e−7 | 2.40e−9 | 1.62e−11 | 4.75e−8 | not applicable |

Classifications agree at all three tolerances. The larger crossing-time
sensitivity at 35.22 degrees is expected near a separatrix; its classification
still has a resolved crossing well before 900. Main-window maximum component
state changes are 1.02e−6, 1.60e−5, and 4.05e−7, respectively.

The finite-window experiment shows that 35 degrees is unresolved at t=225
but escapes by 450; 35.22 degrees is unresolved even at 450 but escapes by
900. The 36-degree orbit has one reflection by 450, two by 900, and three
by 1800, with no barrier crossing. Its largest excursion changes from
53.3005969 (450) to 53.3006179 (900), then stays at that value through 1800.
These observations support finite-time bounce classification, not a proof
of permanent full-orbit confinement. Exactly critical or arbitrarily close
separatrix orbits cannot be classified for all time by any finite run.

The extended t=3600 integrations resolve an operational transition between
**35.222968750 degrees (escaped)** and **35.223046875 degrees (bouncing
observed)**. The bracket width is 7.8125e−5 degrees. Its midpoint is
35.2230078125 degrees, with a half-bracket resolution of 3.90625e−5 degrees;
this is a parameter-resolution interval, not a statistical confidence interval.
The midpoint lies 0.0413818703 degrees below the leading 35.2643896828-degree
prediction. In contrast, the conservative 900-time evidence bracket extends
from 35.22 to 36 degrees because the intermediate cases have not completed
both opposite-side reversals. Its midpoint must not be interpreted as a
precisely measured physical critical angle.

The longer-time shift is consistent with finite-gyroradius corrections and
the difference between particle-axis and guiding-center-axis initialization;
the amplitude sensitivity experiment above directly demonstrates the latter.
Leading loss-cone theory assumes an adiabatic invariant and an on-axis
guiding-center field ratio. The full orbit satisfies neither assumption
exactly. The calculation does not isolate a unique higher-order correction
or establish permanent confinement of every angle above the finite-time
bracket. The much coarser 35-degree/36-degree reference remains valid.

Both extended endpoints retain their classifications at rtol=1e−10 and
1e−12 (atol=1e−12 and 1e−14). The tighter lower endpoint crosses its first
barrier at t=771.7484264; its first-crossing time changes by 0.0071453 between
those tolerances. The upper endpoint has two opposite-side reversals and no
crossing through 3600, with maximum excursion 62.7571871. Maximum relative
K error among all tight extended-boundary runs is 5.64e−11; the two loose
endpoint runs reach about 9.74e−9.

Near-separatrix full trajectories are much more sensitive than their energy
or classification: the largest last-two componentwise state differences over
3600 are 0.02342 (lower endpoint) and 0.01631 (upper endpoint), while maximum
excursions change by 1.779e−4 and 7.163e−6, respectively. Long gyrophase and
transit-time differences can grow despite small energy errors. Thus the
bracket's stable endpoint decisions and small excursion changes do **not**
establish pointwise convergence of the entire long orbit to the same number
of digits. These observed tolerance changes, and the finite observation
window, limit the precision that should be claimed in a presentation.

## Task 5: ideal Penning trap

This independently developed implementation is `05_penning_trap_bhavya.py`.
Matthew-Lindsay's `05_penning_trap.py` from main is preserved unchanged so both
can be compared during review. All outputs from this implementation use the
`task5_bhavya_` prefix in `figures/` and `results/`; it does not write the
partner's script or generic Task 5 output paths. The validation suite imports
the Bhavya implementation explicitly.

`05_penning_trap_bhavya.py` uses q=m=B0=1 and Ω=qB0/m=1. The prescribed potential,
its gradient, and the conserved full Hamiltonian are

\[
\phi=\frac{m\omega_z^2}{2q}\left[z^2-\frac{x^2+y^2}{2}\right],\qquad
\mathbf E=\frac{m\omega_z^2}{2q}(x,y,-2z),\qquad
H=\frac m2v^2+q\phi.
\]

The acceleration components are

\[
\dot v_x=\Omega v_y+\omega_z^2x/2,\quad
\dot v_y=-\Omega v_x+\omega_z^2y/2,\quad
\dot v_z=-\omega_z^2z.
\]

Both radial modes and the axial mode are excited by the same initial state
`(x,y,z,vx,vy,vz)=(1,0,0.3,0,−0.35,0.07)` for each prescribed ratio
ωz/Ω=0.1, 0.2, 0.4, 0.6. The initial H values are 0.06165, 0.0555, 0.0309,
and −0.0101, respectively. Negative total H is allowed because the radial
quadrupole potential contributes negative energy; it does not imply unstable
motion. K alone is not conserved.

Writing w=x+iy gives `w''+iΩw'−ωz²w/2=0`, so

\[
w=A_+e^{-i\omega_+t}+A_-e^{-i\omega_-t},\quad
\omega_\pm=\frac{\Omega\pm\sqrt{\Omega^2-2\omega_z^2}}2,\quad
\omega_+^2+\omega_-^2+\omega_z^2=\Omega^2.
\]

Here ω+ is the **modified cyclotron** mode and ω− is the **magnetron** mode;
their quoted frequencies are positive angular-frequency magnitudes. For
positive qB both complex phases have negative slopes: both circles rotate
clockwise viewed along +z. The magnetron's negative mode energy must not be
confused with a negative quoted frequency or an opposite rotation direction.

The six-dimensional constant matrix is constructed directly from the force
components. Its eigenvalues are computed numerically, and the positive
imaginary parts provide an independent set of mode frequencies. The radial
orbit is then separated **using those numerical eigenfrequencies**, not the
analytical roots:

\[
w_+=\frac{i\dot w-\omega_-w}{\omega_+-\omega_-},\qquad
w_-=\frac{\omega_+w-i\dot w}{\omega_+-\omega_-}.
\]

Unwrapped complex-phase fits of w+, w−, and `z+i vz/ωz_eigen` determine signed
orbit frequencies. This is a trajectory validation in addition to the
matrix-eigenvalue comparison. The projection is tailored to this ideal linear
trap; it is not a general-purpose spectrum estimator for nonlinear fields.
Independently, `scipy.linalg.expm(A*t)` checks 101 full-state samples without
using an analytical orbit fit.

Each ratio is integrated at all three tolerances for
`max(4π/ω−,200π/ω+)`: at least two magnetron periods and 100 modified-cyclotron
periods. Durations are 2500.644284, 641.414186, 688.711231, and 821.787821,
respectively. The output spacing is no greater than 0.08, resolving all modes;
adaptive internal solver steps are independent of that spacing.

| ωz/Ω | Exact ω− | Orbit ω− | Exact ω+ | Orbit ω+ | E×B estimate | Relative estimate error |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.005025253169417 | 0.005025253169417 | 0.994974746830583 | 0.994974746830488 | 0.005 | 0.502525% |
| 0.2 | 0.020416847668728 | 0.020416847668728 | 0.979583152331272 | 0.979583152331180 | 0.020 | 2.041685% |
| 0.4 | 0.087689437438234 | 0.087689437438235 | 0.912310562561766 | 0.912310562561684 | 0.080 | 8.768944% |
| 0.6 | 0.235424868893541 | 0.235424868893541 | 0.764575131106459 | 0.764575131106366 | 0.180 | 23.542487% |

Axial orbit fits yield 0.09999999999999962, 0.20000000000000026,
0.40000000000000474, and 0.5999999999999883. Maximum eigenfrequency relative
error is 1.11e−15, and maximum orbit-frequency relative error is 1.22e−13.
The largest eigenvalue-based frequency-identity residual is 1.33e−15; the
largest orbit-based residual is 1.91e−13, in normalized frequency squared.

The guiding-center estimate is `ω−_GC=ωz²/(2Ω)`. Reported approximation error
is `(ω−_exact−ω−_GC)/ω−_exact`, not an error divided by the approximation.
It underestimates the exact magnetron frequency. The values above reproduce
the professor's approximately 0.5%, 2%, 9%, and 24%; this approximation error
is much larger than numerical integration error.
As an additional algebraic check, ω+ω−=ωz²/2 makes this relative approximation
error equal to ω−/Ω for the ideal trap.

Bounded generic linear motion requires **Ω²>2ωz²**. Numerical eigenvalues
at ratios 0.6 and 0.7 have real parts below 3e−16, while unstable ratios
0.72 and 0.8 have growth rates 0.0959166305 and 0.2645751311. At equality
the repeated radial eigenvalue is defective: geometric multiplicity 1,
algebraic multiplicity 2. The radial propagator norm grows from 12.5795 at
t=10 to 125.008 at t=100, demonstrating secular growth. The roughly 1e−9
real parts returned by floating-point eigenanalysis at the repeated root are
ill-conditioned numerical splitting, not a resolved physical growth rate.
Thus equality is not treated as stable merely because the exact roots are real.

| ωz/Ω | Max absolute H error | Max error / (K0+\|qφ0\|) | Max error / \|H0\| | Last-two maximum frequency change |
|---|---:|---:|---:|---:|
| 0.1 | 2.20e−12 | 3.35e−11 | 3.57e−11 | 9.32e−12 |
| 0.2 | 7.55e−13 | 1.05e−11 | 1.36e−11 | 9.30e−12 |
| 0.4 | 5.74e−13 | 5.95e−12 | 1.86e−11 | 9.28e−12 |
| 0.6 | 2.67e−13 | 1.94e−12 | 2.65e−11 | 9.58e−12 |

Hamiltonian plots normalize by the positive initial scale K0+|qφ0|, avoiding
division by a small signed H0. Both that scaled error and relative-to-|H0|
error are saved in JSON. Tightest full-state errors against the matrix
exponential are 7.87e−11, 1.99e−11, 1.97e−11, and 1.62e−11. Corresponding
last-two-tolerance state changes are 8.18e−9, 2.06e−9, 2.04e−9, and 1.71e−9.

For example, the ratio-0.1 maximum scaled Hamiltonian error falls from
1.012e−6 to 5.706e−9 to 3.346e−11 as rtol tightens; maximum relative
frequency error falls from 9.707e−10 to 9.467e−12 to 9.615e−14.
All four ratios converge similarly. Changes between tolerances are empirical
estimates, not rigorous uncertainty bounds. Phase fitting over many cycles
can measure frequencies more accurately than instantaneous orbit positions.

## New outputs, captions, and automated validation (Tasks 3 and 5)

The same shared `save_figure` function produces vector PDF and 350-DPI PNG for
each new stem below. Existing Task 1–2 figures and numerical files are not
regenerated or overwritten by the new scripts.

| Figure stem in `figures/` | Scientific caption |
|---|---|
| `task3_representative_orbits` | Axial motion and meridional projections of the 35-degree escaping orbit and 36-degree bouncing orbit over t=900. Dashed horizontal lines mark the two adjacent maximum-field planes bounding the initial cell. |
| `task3_transverse_orbits` | Cartesian transverse projections of the same full Lorentz trajectories, with equal x and y scales. Fast gyromotion and the finite offset of the local guiding center are resolved. |
| `task3_loss_cone_scan` | Largest absolute displacement from the starting minimum versus launch pitch. Markers distinguish a measured barrier crossing, observed opposite-side reversals, and unresolved 900-time outcomes. The dashed analytical critical angle and a separate t=3600 numerical bracket are overlaid. The right panel magnifies the fine scan. |
| `task3_convergence` | Kinetic-energy conservation and excursion changes under tolerance refinement for the reference angles and endpoints of the conservative 900-time observation bracket. Differences are measured relative to the tightest run, not to leading guiding-center theory. |
| `task3_duration_sensitivity` | Maximum excursion at nested observation windows 225, 450, 900, and 1800. Passing trajectories continue across successive cells; a shorter observation can miss escape or fail to contain a complete bounce. |
| `task5_bhavya_trajectories` | Radial full-orbit projection and axial oscillation at ωz/Ω=0.4. The mixed initial condition excites the magnetron, modified cyclotron, and axial modes. |
| `task5_bhavya_radial_modes` | Total radial x motion and its numerically projected magnetron and modified-cyclotron contributions. The displayed interval covers two slow-mode periods; the integration is longer. |
| `task5_bhavya_frequencies_and_drift_error` | Exact frequency branches and independent orbit phase fits for all four prescribed ratios, alongside the increasing relative error of the E×B magnetron approximation. Circles coincide with theory to graphical precision. |
| `task5_bhavya_conservation_convergence` | Full-Hamiltonian error along each tight orbit, plotted against fractional run duration, and maximum scaled error versus relative tolerance. Including qφ is essential; K alone varies physically. |
| `task5_bhavya_frequency_convergence` | Maximum relative error across all three orbit-extracted frequencies versus solver tolerance. All ratios converge despite the much larger guiding-center approximation error. |

New numerical files in `results/`:

- `task3_summary.json`: primary scans, events, local centers, conservation,
  finite-duration comparisons, launch sensitivity, and numerical brackets.
- `task3_extended_boundary.json`: extended-time bracket measurements and
  endpoint tolerance comparisons, also included in the main summary.
- `task3_scan.csv`: pitch, duration, excursion, energy error, explicit escape
  and bounce flags, first barrier time, and reversal count. Both flags zero
  means unresolved; an empty CSV barrier time is encoded as NaN, not zero.
- `task3_convergence.csv`: reference/boundary angles at all three tolerances.
- `task3_orbit_35.csv`, `task3_orbit_36.csv`: tight full state trajectories.
- `task5_bhavya_summary.json`: modes, signed phase slopes, identity residuals,
  Hamiltonian diagnostics, matrix-exponential errors, and stability checks.
- `task5_bhavya_modes.csv`, `task5_bhavya_convergence.csv`: directly inspectable comparisons.
- `task5_bhavya_orbit_0.1.csv`, `task5_bhavya_orbit_0.2.csv`, `task5_bhavya_orbit_0.4.csv`,
  `task5_bhavya_orbit_0.6.csv`: tight trajectories, K, qφ, H, and complex radial-mode
  components. Frequencies are angular frequencies, not cycles per unit time.

To repeat only the extended mirror boundary after an existing full scan:

```powershell
python Simulations/03_loss_cone.py --extend-saved
```

This recomputes integrations for the extended boundary and refreshes its JSON
files and scan figure; it does not substitute saved measurements for new ones.
The full reproduction commands at the top rerun everything from parameters.

`validate_tasks_3_5.py` adds six tests: mirror vacuum geometry and axis limits;
pitch and center conventions; non-escape classified as unresolved on a short
run (including the no-turning-event case); electric field checked by numerical
differentiation of φ with positive and negative charges; stability and the
frequency identity from independent NumPy eigenvalues; and validation of saved
scientific results and all ten new PDF/PNG pairs. The existing `validate.py`
continues to check Tasks 1–2 unchanged. Scientific assertions inside the new
scripts additionally check conservation and convergence of the measured cases.

These additions also used OpenAI Codex for code writing, debugging, physical
checks, and documentation. All listed simulated results come from executed
integrations or numerical matrix calculations. The professor's approximate
benchmarks are comparisons, not substituted outputs. Task 4 and the partner's
theoretical material are untouched. The local branch `simulations-tasks-3-5`
was created from synchronized main. Its pull request is intended for independent
review of the equations, numerical methods, physical accuracy, figures, and
benchmark comparisons before merging. Merging requires separate authorization.
