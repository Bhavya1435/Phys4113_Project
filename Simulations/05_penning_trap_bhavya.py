"""Task 5: ideal Penning trap, signed orbit modes and Hamiltonian checks."""
import numpy as np
from scipy.linalg import eigvals, expm
from common import ROOT, TOLERANCES, integrate, style, save_figure, save_json, plt

RATIOS = (.1, .2, .4, .6)
Q = M = B0 = 1.
OMEGA = Q*B0/M

def magnetic(t, x):
    return np.broadcast_to([0., 0., B0], np.asarray(x).shape)

def electric_field(wz, q=Q, m=M):
    def electric(t, x):
        return m*wz*wz/(2*q)*np.asarray(x)*np.array([1., 1., -2.])
    return electric

def potential(x, wz, q=Q, m=M):
    x = np.asarray(x)
    return m*wz*wz/(2*q)*(x[..., 2]**2-(x[..., 0]**2+x[..., 1]**2)/2)

def system_matrix(wz, omega=OMEGA):
    a = np.zeros((6, 6))
    a[:3, 3:] = np.eye(3)
    a[3:, :3] = np.diag([wz*wz/2, wz*wz/2, -wz*wz])
    a[3, 4], a[4, 3] = omega, -omega
    return a

def exact_frequencies(wz):
    gap = np.sqrt(OMEGA**2-2*wz*wz)
    wp = (OMEGA+gap)/2
    # Algebraically equivalent to (Omega-gap)/2, without cancellation.
    wm = wz*wz/(OMEGA+gap)
    return np.array([wm, wz, wp])

def phase_frequency(t, amplitude):
    """Signed phase slope, with a free intercept, from a complex time series."""
    return float(np.polyfit(t, np.unwrap(np.angle(amplitude)), 1)[0])

def extract_modes(t, states, omega_plus, omega_minus):
    """Project w=x+iy using numerically determined eigenfrequencies.

    Both radial modes are exp(-i omega t) for positive qB, even though
    the magnetron has negative mode energy. Input frequencies come from
    matrix eigenvalues, never from the analytical formulas in this check.
    """
    w = states[:, 0]+1j*states[:, 1]
    wd = states[:, 3]+1j*states[:, 4]
    fast = (1j*wd-omega_minus*w)/(omega_plus-omega_minus)
    slow = (omega_plus*w-1j*wd)/(omega_plus-omega_minus)
    return slow, fast

def plot_frequency_convergence(output):
    fig, ax = plt.subplots(figsize=(9, 5))
    for case in output['cases']:
        ax.loglog([r['rtol'] for r in case['runs']],
                  [r['max_relative_frequency_error'] for r in case['runs']], 'o-', label=f"{case['ratio']:.1f}")
    ax.set(xlabel='Relative solver tolerance', ylabel='Maximum relative\nfrequency error')
    ax.legend(title=r'$\omega_z/\Omega$')
    save_figure(fig, 'task5_bhavya_frequency_convergence')

def run():
    style()
    (ROOT/'results').mkdir(exist_ok=True)
    output = {'parameters': dict(q=Q, m=M, B0=B0, omega=OMEGA,
                                initial_state=[1., 0., .3, 0., -.35, .07]), 'cases': []}
    initial = np.array(output['parameters']['initial_state'])
    mode_rows, convergence_rows = [], []
    fig_orbit, orbit_axes = plt.subplots(1, 2, figsize=(13, 5))
    fig_radial, radial_axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig_cons, cons_axes = plt.subplots(1, 2, figsize=(13, 5))
    for ratio in RATIOS:
        wz = ratio*OMEGA
        theory = exact_frequencies(wz)
        a = system_matrix(wz)
        eigen = eigvals(a)
        numerical = np.sort(eigen.imag[eigen.imag > 1e-8])
        assert len(numerical) == 3
        wm, wz_eig, wp = numerical
        # At least two slow-mode periods, and at least 100 fast-mode periods.
        duration = max(4*np.pi/wm, 200*np.pi/wp)
        t = np.linspace(0., duration, int(np.ceil(duration/.08))+1)
        all_states, runs = [], []
        for rtol, atol in TOLERANCES:
            sol = integrate(initial, duration, magnetic, electric_field(wz), rtol, atol)
            states = sol.sol(t).T
            slow, fast = extract_modes(t, states, wp, wm)
            axial = states[:, 2]+1j*states[:, 5]/wz_eig
            signed = np.array([phase_frequency(t, slow), phase_frequency(t, axial),
                               phase_frequency(t, fast)])
            measured = -signed
            kinetic = M*np.sum(states[:, 3:]**2, axis=1)/2
            electrostatic = Q*potential(states[:, :3], wz)
            H = kinetic+electrostatic
            # H can be small or negative; normalize by an initial positive
            # energy scale, not a possibly cancelling signed H(0).
            scale = float(kinetic[0]+abs(electrostatic[0]))
            run_data = dict(rtol=rtol, atol=atol, nfev=sol.nfev,
                            signed_phase_slopes=signed.tolist(),
                            measured_frequencies=measured.tolist(),
                            max_relative_frequency_error=float(np.max(abs(measured/theory-1))),
                            frequency_identity_residual=float(np.sum(measured**2)-OMEGA**2),
                            hamiltonian_initial=float(H[0]), energy_scale=scale,
                            max_absolute_H_error=float(np.max(abs(H-H[0]))),
                            max_scaled_H_error=float(np.max(abs(H-H[0]))/scale),
                            max_relative_H_error=float(np.max(abs(H/H[0]-1))))
            runs.append(run_data)
            all_states.append(states)
            convergence_rows.append([ratio, rtol, atol, *measured,
                                     run_data['max_scaled_H_error'], run_data['max_relative_frequency_error']])
        # Independent matrix-exponential values, not a fitted mode solution.
        check_t = np.linspace(0., duration, 101)
        exact = np.array([expm(a*tt)@initial for tt in check_t])
        matrix_error = float(np.max(abs(sol.sol(check_t).T-exact)))
        approximate = wz*wz/(2*OMEGA)
        approx_error = float((theory[0]-approximate)/theory[0])
        case = dict(ratio=ratio, duration=duration, theoretical_frequencies=theory.tolist(),
                    eigen_frequencies=numerical.tolist(),
                    max_eigen_relative_error=float(np.max(abs(numerical/theory-1))),
                    eigen_real_parts=eigen.real.tolist(),
                    eigen_identity_residual=float(np.sum(numerical**2)-OMEGA**2),
                    guiding_center_magnetron=approximate, approximation_relative_error=approx_error,
                    runs=runs, matrix_exponential_max_state_error=matrix_error,
                    convergence=dict(max_state_change_last_two=float(np.max(abs(all_states[-1]-all_states[-2]))),
                                     max_frequency_change_last_two=float(np.max(abs(np.array(runs[-1]['measured_frequencies'])-
                                                                                   runs[-2]['measured_frequencies'])))))
        output['cases'].append(case)
        assert runs[-1]['max_scaled_H_error'] < 1e-8
        assert runs[-1]['max_relative_frequency_error'] < 1e-8
        assert matrix_error < 1e-7
        mode_rows.append([ratio, *theory, *numerical, *runs[-1]['measured_frequencies'], approximate, approx_error])
        np.savetxt(ROOT/'results'/f'task5_bhavya_orbit_{ratio:.1f}.csv',
                   np.column_stack((t, states, kinetic, electrostatic, H,
                                    slow.real, slow.imag, fast.real, fast.imag)), delimiter=',',
                   header='t,x,y,z,vx,vy,vz,K,q_phi,H,slow_x,slow_y,fast_x,fast_y', comments='')
        cons_axes[0].plot(t/duration, (H-H[0])/scale, label=f'{ratio:.1f}')
        cons_axes[1].loglog([r['rtol'] for r in runs],
                            [r['max_scaled_H_error'] for r in runs], 'o-', label=f'{ratio:.1f}')
        if ratio == .4:
            shown = t <= 4*2*np.pi/wm
            orbit_axes[0].plot(states[shown, 0], states[shown, 1], lw=.8)
            orbit_axes[0].set(xlabel='$x$ (normalized)', ylabel='$y$ (normalized)',
                              title=r'Radial orbit: $\omega_z/\Omega=0.4$', aspect='equal')
            orbit_axes[1].plot(t[shown], states[shown, 2])
            orbit_axes[1].set(xlabel='$t$ (normalized)', ylabel='$z$ (normalized)', title='Axial oscillation')
            window = t <= 2*2*np.pi/wm
            radial_axes[0].plot(t[window], states[window, 0], label='Total $x$')
            radial_axes[0].plot(t[window], slow.real[window], label='Magnetron contribution')
            radial_axes[1].plot(t[window], fast.real[window], label='Modified cyclotron contribution')
        print(f'Penning ratio {ratio:.1f}: completed all tolerances', flush=True)
    # Stability is diagnosed from eigenvalues independently on both sides.
    output['stability'] = []
    for ratio in (.6, .7, 1/np.sqrt(2), .72, .8):
        eigen = eigvals(system_matrix(ratio*OMEGA))
        output['stability'].append(dict(ratio=float(ratio),
                                       max_real_part=float(np.max(eigen.real)),
                                       min_real_part=float(np.min(eigen.real))))
    output['stability_note'] = 'Equality is a repeated-root boundary with secular solutions; bounded generic motion requires strict inequality.'
    critical_matrix = system_matrix(OMEGA/np.sqrt(2))
    radial_indices = [0, 1, 3, 4]
    critical_radial = critical_matrix[np.ix_(radial_indices, radial_indices)]
    output['marginal_stability'] = dict(
        radial_eigenspace_dimension=int(4-np.linalg.matrix_rank(critical_radial+.5j*OMEGA*np.eye(4), tol=1e-12)),
        radial_algebraic_multiplicity=2,
        propagator_norm_t10=float(np.linalg.norm(expm(10*critical_radial), 2)),
        propagator_norm_t100=float(np.linalg.norm(expm(100*critical_radial), 2)))
    orbit_axes[1].ticklabel_format(axis='y', style='plain')
    save_figure(fig_orbit, 'task5_bhavya_trajectories')
    for ax in radial_axes:
        ax.set(ylabel='$x$ contribution (normalized)')
        ax.legend()
    radial_axes[-1].set(xlabel='$t$ (normalized)')
    save_figure(fig_radial, 'task5_bhavya_radial_modes')
    cons_axes[0].set(xlabel='Fraction of integration duration', ylabel=r'$(H-H_0)/(K_0+|q\phi_0|)$')
    cons_axes[1].set(xlabel='Relative solver tolerance', ylabel='Maximum scaled Hamiltonian error')
    for ax in cons_axes:
        ax.legend(title=r'$\omega_z/\Omega$')
    save_figure(fig_cons, 'task5_bhavya_conservation_convergence')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data = np.array(mode_rows)
    for j, label in enumerate((r'$\omega_-$', r'$\omega_z$', r'$\omega_+$')):
        line, = axes[0].plot(data[:, 0], data[:, 1+j], '-', label=label+' exact')
        axes[0].plot(data[:, 0], data[:, 7+j], 'o', mfc='none', color=line.get_color())
    axes[0].set(xlabel=r'$\omega_z/\Omega$', ylabel=r'Angular frequency / $\Omega$')
    axes[0].legend()
    axes[0].set_title('Lines: exact; open circles: orbit phase fits')
    axes[1].plot(data[:, 0], 100*data[:, -1], 'o-')
    axes[1].set(xlabel=r'$\omega_z/\Omega$', ylabel='Magnetron approximation error (%)')
    save_figure(fig, 'task5_bhavya_frequencies_and_drift_error')
    plot_frequency_convergence(output)
    np.savetxt(ROOT/'results'/'task5_bhavya_modes.csv', data, delimiter=',',
               header='ratio,exact_wminus,exact_wz,exact_wplus,eigen_wminus,eigen_wz,eigen_wplus,orbit_wminus,orbit_wz,orbit_wplus,gc_wminus,gc_relative_error', comments='')
    np.savetxt(ROOT/'results'/'task5_bhavya_convergence.csv', convergence_rows, delimiter=',',
               header='ratio,rtol,atol,orbit_wminus,orbit_wz,orbit_wplus,scaled_H_error,max_relative_frequency_error', comments='')
    save_json('task5_bhavya_summary.json', output)

if __name__ == '__main__':
    run()
