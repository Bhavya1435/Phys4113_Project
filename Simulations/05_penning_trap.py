import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

big_omega = 1
ratios  = [0.1, 0.2, 0.4, 0.6]

def eqns_of_motion(t, state, w_z):
    x, y, z, v_x, v_y, v_z = state
    a_x = big_omega * v_y + (w_z**2 / 2) * x
    a_y = -big_omega * v_x + (w_z**2 / 2) * y
    a_z = -w_z**2 *z
    return [v_x, v_y, v_z, a_x, a_y, a_z]

start  = [3, 0, 1, 0, 0.3, 0]

formula_minus = []
eigen_minus = []
orbit_minus = []
ExB_minus = []
errors = []

for ratio in ratios:
    w_z = ratio * big_omega
    print("=" * 60)
    print(f"w_z / big_omega = {ratio}")
    print("=" * 60)

    # DEFINE FORMULAS FOR w_plus AND w_minus
    w_plus_formula = 0.5 * (big_omega + np.sqrt(big_omega**2 - 2 * w_z**2))
    w_minus_formula = 0.5 * (big_omega - np.sqrt(big_omega**2 - 2 * w_z**2))

    # DEFINE EIGENVALUES FOR w_plus AND w_minus
    M = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [w_z**2/2, 0, 0, 0, big_omega, 0],
        [0, w_z**2/2, 0, -big_omega, 0, 0],
        [0, 0, -w_z**2, 0, 0, 0]
    ])

    eigenvalues = np.linalg.eigvals(M)
    frequencies = np.sort(eigenvalues.imag[eigenvalues.imag > 0])
    w_minus_eigen = frequencies[0]
    w_z_eigen = frequencies[1]
    w_plus_eigen = frequencies[2]

    # SIMULATE ORBIT USING solve_ivp
    t_end = 40 * 2 * np.pi / w_minus_formula
    n_points = int(40 * t_end / (2 *np.pi / w_plus_formula))
    times = np.linspace(0, t_end, n_points)

    solution = solve_ivp(eqns_of_motion, [0, t_end], start, args=(w_z,), t_eval=times, method='DOP853', rtol=1e-11, atol=1e-12)

    x = solution.y[0]
    z = solution.y[2]

    delta_t = times[1] - times[0]
    n_fft = 32 * len(x)
    spectrum_x = np.abs(np.fft.rfft(x - np.mean(x), n=n_fft))
    spectrum_z = np.abs(np.fft.rfft(z - np.mean(z), n=n_fft))
    w_grid = 2 * np.pi * np.fft.rfftfreq(n_fft, delta_t)

    low = w_grid < big_omega / 2
    high = w_grid > big_omega / 2
    w_minus_orbit = w_grid[low][np.argmax(spectrum_x[low])]
    w_plus_orbit = w_grid[high][np.argmax(spectrum_x[high])]
    w_z_orbit = w_grid[np.argmax(spectrum_z)]


    # EXB ESTIMATE OF THE MAGNETRON FREQUENCY
    w_minus_EXB = w_z**2 / (2 * big_omega)
    error_percent = (w_minus_formula - w_minus_EXB) / w_minus_formula * 100
    eqn_five = w_plus_eigen**2 + w_minus_eigen**2 + w_z_eigen**2

    print("                     formula     eigenvalues  simulation")
    print(f"  w- (magnetron):    {w_minus_formula:.6f}     {w_minus_eigen:.6f}     {w_minus_orbit:.6f}")
    print(f"  w_z (axial):       {w_z:.6f}     {w_z_eigen:.6f}     {w_z_orbit:.6f}")
    print(f"  w_(+) (cyclotron): {w_plus_formula:.6f}     {w_plus_eigen:.6f}     {w_plus_orbit:.6f})")
    print(" ")
    print(f"  w_(+)^2 + w_(-)^2 + w_z^2 = {eqn_five:.10f}") #shoule be like 1
    print(f"  E x B estimate of w_(-) = {w_minus_EXB:.6f}")
    print(f"  Error of E x B estimate = {error_percent:.2f} %")

    formula_minus.append(w_minus_formula)
    eigen_minus.append(w_minus_eigen)
    orbit_minus.append(w_minus_orbit)
    ExB_minus.append(w_minus_EXB)
    errors.append(error_percent)

#   SATBILITY ANALYSIS    
for ratio in [0.7, 0.72]:
    w_z = ratio * big_omega
    M = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [w_z**2/2, 0, 0, 0, big_omega, 0],
        [0, w_z**2/2, 0, -big_omega, 0, 0], 
        [0, 0, -w_z**2, 0, 0, 0],
    ])
    largest_real_part = np.max(np.linalg.eigvals(M).real)
    if largest_real_part > 1e-9:
        we_have = "unstable and so the particle will escape"
    else:
        we_have = "stable and so the particle will stay trapped"
    print(f"w_z / big_omega = {ratio}: largest real part = {largest_real_part:.6f} so we have {we_have}")

# Work on figures.