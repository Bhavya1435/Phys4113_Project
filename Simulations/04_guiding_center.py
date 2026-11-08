"""Task 4: full Lorentz orbit versus the specified first-order guiding center."""
import csv
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from common import (ROOT, TOLERANCES, bottle_field, integrate, diagnostics,
                    style, save_figure, save_json, plt)

Q = M = B0 = 1.
L, R0, VPERP, UPAR, DURATION = 10., 3., .5, .2, 700.
X0 = np.array([R0, 0., 0.])
INITIAL_PHASE = np.pi/2

def magnetic(t, position):
    return bottle_field(t, position, B0=B0, L=L)

def field_jacobian(position):
    p = np.asarray(position, dtype=float)
    x, y, z = np.moveaxis(p, -1, 0)
    jac = np.zeros(p.shape[:-1]+(3, 3))
    factor = B0/L**2
    jac[..., 0, 0] = jac[..., 1, 1] = -factor*z
    jac[..., 0, 2] = jac[..., 2, 0] = -factor*x
    jac[..., 1, 2] = jac[..., 2, 1] = -factor*y
    jac[..., 2, 2] = 2*factor*z
    return jac

def geometry(position):
    """Return B, b, grad|B| and kappa using the full vector-field Jacobian.

    J_ij = dB_i/dx_j; grad|B| = J.T b;
    Db = (I-b b.T) J/|B|; kappa = Db b.
    Supports a single point or an array of points (..., 3).
    """
    p = np.asarray(position, dtype=float)
    field = magnetic(0., p)
    strength = np.linalg.norm(field, axis=-1)
    b = field/strength[..., None]
    jac = field_jacobian(p)
    grad = np.einsum('...ij,...i->...j', jac, b)
    jb = np.einsum('...ij,...j->...i', jac, b)
    curvature = (jb-b*np.sum(b*jb, axis=-1)[..., None])/strength[..., None]
    return strength, b, grad, curvature

def initial_state(phase=INITIAL_PHASE, parallel_speed=UPAR):
    """Invert the local first-order center map, with X(0)=(3,0,0).

    At z=0 the full field is vertical. The prescribed perpendicular and
    parallel speeds therefore hold at the particle position exactly.
    Phase is the Cartesian perpendicular velocity angle at t=0.
    """
    vx, vy = VPERP*np.cos(phase), VPERP*np.sin(phase)
    def position(B):
        return X0+M/(Q*B)*np.array([-vy, vx, 0.])
    def residual(B):
        return B-float(magnetic(0., position(B))[2])
    strength = brentq(residual, .8, 1.05, xtol=1e-14)
    return np.r_[position(strength), vx, vy, parallel_speed]

def center_rhs(state, mu):
    B, b, grad, curvature = geometry(state[:3])
    drift = np.cross(b, mu*grad+M*state[3]**2*curvature)/(Q*B)
    return np.r_[state[3]*b+drift, -mu*np.dot(b, grad)/M]

def integrate_center(mu, rtol=1e-12, atol=1e-14, duration=DURATION, initial_u=UPAR):
    """Equation (3) exactly as supplied, for static E=0; state=(X,Y,Z,u)."""
    def rhs(t, state):
        return center_rhs(state, mu)
    solution = solve_ivp(rhs, (0., duration), np.r_[X0, initial_u],
                         method='DOP853', rtol=rtol, atol=atol, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution

def orbit_diagnostics(t, states):
    energy, mu, center = diagnostics(states, magnetic, m=M, q=Q)
    B, b, grad, curvature = geometry(states[:, :3])
    u = np.sum(states[:, 3:]*b, axis=1)
    vp = states[:, 3:]-u[:, None]*b
    radius = np.hypot(center[:, 0], center[:, 1])
    er = np.column_stack((center[:, 0]/radius, center[:, 1]/radius, np.zeros_like(radius)))
    # Anchor the gyroframe to the *center's* slow azimuth, not the rapidly
    # changing particle azimuth. Project onto the plane perpendicular to B(x).
    e1 = er-np.sum(er*b, axis=1)[:, None]*b
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(b, e1)
    phase = np.unwrap(np.arctan2(np.sum(vp*e2, axis=1), np.sum(vp*e1, axis=1)))
    progress = phase[0]-phase
    if not np.all(np.diff(progress) > 0):
        raise RuntimeError('Gyrophase is not monotone; the averaging convention must be reconsidered')
    return dict(energy=energy, mu=mu, center=center, B=B, b=b, u=u, vperp=vp,
                phase=phase, progress=progress, rho=states[:, :3]-center,
                gyroradius=np.linalg.norm(states[:, :3]-center, axis=1))

def gyro_averages(t, diagnostic):
    """Empirical gyro-time mean and explicitly separate phase/window alternatives.

    The primary constant is the whole-orbit time integral divided by elapsed
    time. This averages many gyroperiods and bounces, as an empirical input
    to the requested comparison. It is not an exact Lie-transformed invariant.
    Complete-cycle time means and uniform-phase means quantify ambiguity at
    higher order. Cubic interpolants integrate the resolved samples directly.
    """
    progress = diagnostic['progress']
    count = int(np.floor(progress[-1]/(2*np.pi)))
    edges = 2*np.pi*np.arange(count+1)
    phase_spline = CubicSpline(progress, diagnostic['mu'])
    cycle_phase_mu = np.diff(phase_spline.antiderivative()(edges))/(2*np.pi)
    cycle_times = CubicSpline(progress, t)(edges)
    time_spline = CubicSpline(t, diagnostic['mu'])
    time_integrals = time_spline.antiderivative()(cycle_times)
    cycle_time_mu = np.diff(time_integrals)/np.diff(cycle_times)
    primary = float(time_spline.integrate(t[0], t[-1])/(t[-1]-t[0]))
    complete = float((time_integrals[-1]-time_integrals[0])/(cycle_times[-1]-cycle_times[0]))
    alternatives = dict(whole_orbit_time=primary, complete_cycles_time=complete,
                        whole_orbit_uniform_phase=float(np.mean(cycle_phase_mu)))
    for count_used in (1, 4, 8):
        alternatives[f'first_{count_used}_cycles_time'] = float(
            (time_integrals[count_used]-time_integrals[0])/(cycle_times[count_used]-cycle_times[0]))
    return primary, alternatives, cycle_time_mu, cycle_phase_mu, cycle_times, edges

def azimuth(position):
    angle = np.unwrap(np.arctan2(position[:, 1], position[:, 0]))
    return angle-angle[0]

def center_metrics(t, center, mu, rtol=1e-12, atol=1e-14):
    sol = integrate_center(mu, rtol, atol)
    prediction = sol.sol(t).T
    difference = np.linalg.norm(prediction[:, :3]-center, axis=1)
    B = geometry(prediction[:, :3])[0]
    H = M*prediction[:, 3]**2/2+mu*B
    return dict(mu=float(mu), max_position_error=float(np.max(difference)),
                rms_position_error=float(np.sqrt(np.mean(difference**2))),
                final_azimuth=float(azimuth(prediction[:, :3])[-1]),
                gc_max_relative_energy_error=float(np.max(abs(H/H[0]-1))),
                nfev=sol.nfev), prediction, difference

def correction_metrics(t, diagnostic, mu):
    """Test Eq. (4) as written, without fitting mu or rescaling its RHS.

    The extra curvature projection is a diagnostic only, not a replacement
    for the professor's Eq. (4). It helps locate a systematic residual.
    """
    B, b, grad, curvature = geometry(diagnostic['center'])
    lhs = diagnostic['mu']/mu-1
    rhs = -np.sum(diagnostic['rho']*grad, axis=1)/B
    extra = -M*diagnostic['u']**2*np.sum(diagnostic['rho']*curvature, axis=1)/(mu*B)
    particle_B, _, particle_grad, _ = geometry(diagnostic['center']+diagnostic['rho'])
    particle_rhs = -np.sum(diagnostic['rho']*particle_grad, axis=1)/particle_B
    residual = lhs-rhs
    slope, intercept = np.polyfit(rhs, lhs, 1)
    psi = diagnostic['progress']
    cycles = int(psi[-1]/(2*np.pi))
    uniform_psi = np.linspace(0., 2*np.pi*cycles, 128*cycles, endpoint=False)
    def harmonic(signal):
        sampled = CubicSpline(psi, signal)(uniform_psi)
        a = 2*np.mean(sampled*np.cos(uniform_psi))
        b = 2*np.mean(sampled*np.sin(uniform_psi))
        return float(np.hypot(a, b)), float(np.arctan2(b, a))
    amp_lhs, phase_lhs = harmonic(lhs)
    amp_rhs, phase_rhs = harmonic(rhs)
    lag = float(np.angle(np.exp(1j*(phase_lhs-phase_rhs))))
    metrics = dict(correlation=float(np.corrcoef(lhs, rhs)[0, 1]),
                   fitted_slope=float(slope), fitted_intercept=float(intercept),
                   lhs_peak_to_peak=float(np.ptp(lhs)), rhs_peak_to_peak=float(np.ptp(rhs)),
                   rms_residual=float(np.sqrt(np.mean(residual**2))),
                   max_absolute_residual=float(np.max(abs(residual))),
                   mean_residual=float(np.mean(residual)),
                   lhs_first_harmonic_amplitude=amp_lhs, rhs_first_harmonic_amplitude=amp_rhs,
                   first_harmonic_amplitude_ratio=amp_lhs/amp_rhs,
                   first_harmonic_phase_difference_rad=lag,
                   rms_residual_with_curvature_diagnostic=float(np.sqrt(np.mean((residual-extra)**2))),
                   rms_rhs_change_particle_vs_center=float(np.sqrt(np.mean((particle_rhs-rhs)**2))))
    return metrics, lhs, rhs, residual, extra

def one_run(phase=INITIAL_PHASE, rtol=1e-12, atol=1e-14, dt=.02):
    t = np.linspace(0., DURATION, int(round(DURATION/dt))+1)
    initial = initial_state(phase)
    solution = integrate(initial, DURATION, magnetic, rtol=rtol, atol=atol, q=Q, m=M)
    states = solution.sol(t).T
    d = orbit_diagnostics(t, states)
    bar, alternatives, cycle_time, cycle_phase, cycle_times, edges = gyro_averages(t, d)
    instantaneous, ci, ei = center_metrics(t, d['center'], d['mu'][0], rtol, atol)
    averaged, ca, ea = center_metrics(t, d['center'], bar, rtol, atol)
    # Compare matched cycle averages of both trajectories, rather than
    # comparing a cycle-averaged point with an instantaneous model position.
    for metrics, prediction in ((instantaneous, ci), (averaged, ca)):
        integrated = CubicSpline(d['progress'], prediction[:, :3]-d['center']).antiderivative()(edges)
        mean_difference = np.diff(integrated, axis=0)/(2*np.pi)
        metrics['max_cycle_mean_position_difference'] = float(np.max(np.linalg.norm(mean_difference, axis=1)))
    eq4, lhs, rhs, residual, extra = correction_metrics(t, d, bar)
    summary = dict(rtol=rtol, atol=atol, initial_phase_rad=float(phase), initial_state=initial.tolist(),
                   initial_center=d['center'][0].tolist(), initial_particle_field=float(d['B'][0]),
                   full_nfev=solution.nfev, mu_initial=float(d['mu'][0]), mu_average=bar,
                   mu_difference_relative_to_average=float((bar-d['mu'][0])/bar),
                   mu_difference_relative_to_initial=float((bar-d['mu'][0])/d['mu'][0]),
                   full_max_relative_energy_error=float(np.max(abs(d['energy']/d['energy'][0]-1))),
                   full_final_azimuth=float(azimuth(d['center'])[-1]),
                   instantaneous=instantaneous, averaged=averaged,
                   error_reduction_factor=instantaneous['max_position_error']/averaged['max_position_error'],
                   equation4=eq4, averaging=dict(alternatives=alternatives, complete_cycles=len(cycle_time),
                                                complete_cycle_end_time=float(cycle_times[-1]),
                                                cycle_time_mean_range=float(np.ptp(cycle_time)),
                                                cycle_phase_mean_range=float(np.ptp(cycle_phase))))
    data = dict(t=t, states=states, diagnostic=d, instant=ci, averaged=ca,
                instant_error=ei, averaged_error=ea, cycle_time_mu=cycle_time,
                cycle_phase_mu=cycle_phase, cycle_edges_time=cycle_times,
                cycle_edges_phase=edges, eq4_lhs=lhs, eq4_rhs=rhs,
                eq4_residual=residual, curvature_diagnostic=extra)
    return summary, data, solution

def run():
    style()
    (ROOT/'results').mkdir(exist_ok=True)
    runs, data_by_tolerance = [], []
    for rtol, atol in TOLERANCES:
        summary, data, solution = one_run(rtol=rtol, atol=atol)
        runs.append(summary)
        data_by_tolerance.append(data)
        print(f'Task 4 tolerance {rtol:g}: mu={summary["mu_average"]:.12f}, '
              f'errors={summary["instantaneous"]["max_position_error"]:.6f}, '
              f'{summary["averaged"]["max_position_error"]:.6f}', flush=True)
    tight, previous = runs[-1], runs[-2]
    data, before = data_by_tolerance[-1], data_by_tolerance[-2]
    t, states, d = data['t'], data['states'], data['diagnostic']
    convergence = dict(max_full_state_change=float(np.max(abs(states-before['states']))),
                       max_extracted_center_change=float(np.max(np.linalg.norm(d['center']-before['diagnostic']['center'], axis=1))),
                       max_instant_gc_state_change=float(np.max(abs(data['instant']-before['instant']))),
                       max_averaged_gc_state_change=float(np.max(abs(data['averaged']-before['averaged']))),
                       mu_average_change=abs(tight['mu_average']-previous['mu_average']),
                       instant_max_error_change=abs(tight['instantaneous']['max_position_error']-previous['instantaneous']['max_position_error']),
                       averaged_max_error_change=abs(tight['averaged']['max_position_error']-previous['averaged']['max_position_error']),
                       full_final_azimuth_change=abs(tight['full_final_azimuth']-previous['full_final_azimuth']))
    # Refine only the observation grid, independently of adaptive solver steps.
    fine_t = np.linspace(0., DURATION, 70001)
    fine_d = orbit_diagnostics(fine_t, solution.sol(fine_t).T)
    fine_average = gyro_averages(fine_t, fine_d)[0]
    output = dict(parameters=dict(q=Q, m=M, B0=B0, L=L, r0=R0, vperp=VPERP, u0=UPAR,
                                  duration=DURATION, output_dt=.02, initial_phase_rad=INITIAL_PHASE,
                                  initial_center=X0.tolist()),
                  runs=runs, convergence=convergence,
                  sampling=dict(refined_dt=.01, refined_mu_average=fine_average,
                                mu_average_change=abs(fine_average-tight['mu_average'])),
                  averaging_sensitivity=[], gyrophase_sensitivity=[], parallel_speed_control=[])
    for name, mu in tight['averaging']['alternatives'].items():
        metrics, _, _ = center_metrics(t, d['center'], mu)
        output['averaging_sensitivity'].append(dict(convention=name, **metrics))
    for phase in (-np.pi/2, 0., np.pi):
        summary, _, _ = one_run(phase=phase, rtol=1e-10, atol=1e-12)
        output['gyrophase_sensitivity'].append(summary)
        print(f'Task 4 gyrophase {phase/np.pi:.2f} pi checked', flush=True)
    # A short auxiliary u0=0 control probes the parallel-motion contribution
    # to the Eq. (4) residual. It does not change the prescribed main run.
    control_t = np.linspace(0., 80., 4001)
    control_sol = integrate(initial_state(parallel_speed=0.), 80., magnetic, rtol=1e-12, atol=1e-14)
    control_d = orbit_diagnostics(control_t, control_sol.sol(control_t).T)
    control_mu = gyro_averages(control_t, control_d)[0]
    metrics = correction_metrics(control_t, control_d, control_mu)[0]
    output['parallel_speed_control'] = dict(initial_u=0., duration=80., mu_average=control_mu, equation4=metrics)
    assert tight['full_max_relative_energy_error'] < 1e-8
    assert tight['instantaneous']['gc_max_relative_energy_error'] < 1e-8
    assert tight['averaged']['gc_max_relative_energy_error'] < 1e-8
    assert convergence['max_full_state_change'] < 1e-5
    assert convergence['averaged_max_error_change'] < 1e-5
    assert tight['error_reduction_factor'] > 5.
    save_outputs(output, data)
    make_figures(output, data)
    save_json('task4_summary.json', output)

def save_outputs(output, data):
    t, s, d = data['t'], data['states'], data['diagnostic']
    np.savetxt(ROOT/'results'/'task4_orbit.csv',
               np.column_stack((t, s, d['center'], d['u'], d['B'], d['mu'], d['energy'],
                                d['phase'], d['progress'], d['gyroradius'], d['rho'])), delimiter=',',
               header='t,x,y,z,vx,vy,vz,X,Y,Z,u,B,mu,K,gyrophase,phase_progress,gyroradius,rho_x,rho_y,rho_z', comments='')
    np.savetxt(ROOT/'results'/'task4_guiding_centers.csv',
               np.column_stack((t, d['center'], data['instant'], data['averaged'],
                                data['instant_error'], data['averaged_error'], azimuth(d['center']),
                                azimuth(data['instant'][:, :3]), azimuth(data['averaged'][:, :3]))), delimiter=',',
               header='t,X_full,Y_full,Z_full,X_initial_mu,Y_initial_mu,Z_initial_mu,u_initial_mu,X_mean_mu,Y_mean_mu,Z_mean_mu,u_mean_mu,error_initial_mu,error_mean_mu,phi_full,phi_initial_mu,phi_mean_mu', comments='')
    np.savetxt(ROOT/'results'/'task4_moment_correction.csv',
               np.column_stack((t, data['eq4_lhs'], data['eq4_rhs'], data['eq4_residual'], data['curvature_diagnostic'])),
               delimiter=',', header='t,mu_ratio_minus_one,minus_rho_dot_gradB_over_B,residual,curvature_diagnostic', comments='')
    np.savetxt(ROOT/'results'/'task4_gyrocycle_means.csv',
               np.column_stack((data['cycle_edges_time'][:-1], data['cycle_edges_time'][1:],
                                data['cycle_time_mu'], data['cycle_phase_mu'])), delimiter=',',
               header='cycle_start,cycle_end,time_mean_mu,uniform_phase_mean_mu', comments='')
    np.savetxt(ROOT/'results'/'task4_convergence.csv',
               [[r['rtol'], r['atol'], r['mu_initial'], r['mu_average'], r['instantaneous']['max_position_error'],
                 r['averaged']['max_position_error'], r['full_final_azimuth'], r['full_max_relative_energy_error'],
                 r['instantaneous']['gc_max_relative_energy_error'], r['averaged']['gc_max_relative_energy_error']]
                for r in output['runs']], delimiter=',',
               header='rtol,atol,mu_initial,mu_average,max_error_initial_mu,max_error_mean_mu,final_phi_full,full_relative_K_error,gc_initial_relative_H_error,gc_mean_relative_H_error', comments='')
    with (ROOT/'results'/'task4_sensitivities.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['averaging_convention', 'mu', 'max_position_error', 'final_azimuth'])
        writer.writerows([[r['convention'], r['mu'], r['max_position_error'], r['final_azimuth']]
                          for r in output['averaging_sensitivity']])

def make_figures(output, data):
    t, s, d = data['t'], data['states'], data['diagnostic']
    ci, ca = data['instant'], data['averaged']
    # Fixed margins avoid a 3-D constrained-layout issue that can clip z labels.
    fig = plt.figure(figsize=(12, 8), layout='none')
    fig.subplots_adjust(left=.02, right=.82, bottom=.08, top=.88)
    ax = fig.add_subplot(projection='3d')
    ax.plot(*s[::5, :3].T, lw=.5, color='gray', alpha=.6, label='Full particle orbit')
    ax.plot(*d['center'][::5].T, lw=1.5, color='tab:blue', label='Extracted center')
    ax.plot(*ci[::5, :3].T, '--', color='tab:orange', label='GC: initial moment')
    ax.plot(*ca[::5, :3].T, ':', color='tab:green', label='GC: time-averaged moment')
    ax.set(xlabel='$x$ (normalized)', ylabel='$y$ (normalized)', zlabel='')
    # A 2-D text artist participates correctly in tight PDF/PNG bounding boxes.
    ax.text2D(1.04, .5, '$z$ (normalized)', transform=ax.transAxes,
              rotation=90, va='center', ha='left', fontsize=17)
    ax.legend(fontsize=10)
    fig.canvas.draw()
    save_figure(fig, 'task4_full_orbit_and_center')
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for position, label, ls in [(d['center'], 'Extracted center', '-'),
                                (ci[:, :3], 'GC: initial moment', '--'),
                                (ca[:, :3], 'GC: time-averaged moment', ':')]:
        axes[0].plot(t, np.hypot(position[:, 0], position[:, 1]), ls, label=label)
        axes[1].plot(t, position[:, 2], ls, label=label)
        axes[2].plot(t, azimuth(position), ls, label=label)
    for ax, label in zip(axes, ('$R$ (normalized)', '$Z$ (normalized)', r'$\varphi-\varphi_0$ (rad)')):
        ax.set(ylabel=label)
    axes[0].legend()
    axes[-1].set(xlabel='$t$ (normalized)')
    save_figure(fig, 'task4_center_coordinates')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, data['instant_error'], color='tab:orange', label='Initial moment')
    axes[0].plot(t, data['averaged_error'], color='tab:green', label='Time-averaged moment')
    axes[1].plot(t, data['averaged_error'], color='tab:green')
    axes[0].set(ylabel='Position error\n(normalized)')
    axes[1].set(xlabel='$t$ (normalized)', ylabel='Averaged-moment error\n(normalized)')
    axes[0].legend()
    save_figure(fig, 'task4_position_errors')
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, azimuth(d['center']), label='Extracted center')
    ax.plot(t, azimuth(ci[:, :3]), '--', label='GC: initial moment')
    ax.plot(t, azimuth(ca[:, :3]), ':', label='GC: time-averaged moment')
    ax.set(xlabel='$t$ (normalized)', ylabel=r'$\varphi-\varphi_0$ (rad)')
    ax.legend()
    save_figure(fig, 'task4_azimuth')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    shown = t <= 45
    axes[0].plot(t[shown], d['mu'][shown], label=r'$\mu_{inst}$')
    for value, label, ls, color in [(output['runs'][-1]['mu_average'], 'Whole-orbit time mean', '--', 'tab:green'),
                                     (d['mu'][0], 'Initial moment', ':', 'tab:orange')]:
        axes[0].axhline(value, ls=ls, label=label, color=color)
    midpoints = (data['cycle_edges_time'][:-1]+data['cycle_edges_time'][1:])/2
    axes[1].plot(midpoints, data['cycle_time_mu'], label='Cycle time means')
    axes[1].plot(midpoints, data['cycle_phase_mu'], label='Cycle uniform-phase means')
    axes[1].axhline(output['runs'][-1]['mu_average'], color='black', ls='--', label='Primary time mean')
    for ax in axes:
        ax.set(xlabel='$t$ (normalized)', ylabel=r'$\mu$ (normalized)')
        ax.legend(fontsize=11)
    save_figure(fig, 'task4_magnetic_moment')
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    shown = t <= 45
    axes[0].plot(t[shown], data['eq4_lhs'][shown], label=r'$\mu_{inst}/\bar\mu-1$')
    axes[0].plot(t[shown], data['eq4_rhs'][shown], '--', label=r'$-\boldsymbol{\rho}\cdot\nabla B/B$')
    axes[1].plot(t[shown], data['eq4_residual'][shown], label='Eq. (4) residual')
    axes[1].plot(t[shown], data['curvature_diagnostic'][shown], '--', label='Curvature projection diagnostic')
    axes[2].plot(t[shown], (data['eq4_residual']-data['curvature_diagnostic'])[shown])
    axes[0].set(ylabel='Relative correction')
    axes[1].set(ylabel='Relative residual')
    axes[2].set(xlabel='$t$ (normalized)', ylabel='Residual after\ncurvature diagnostic')
    axes[0].legend()
    axes[1].legend(fontsize=11)
    save_figure(fig, 'task4_moment_correction')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, key in [('Full orbit', 'full_max_relative_energy_error')]:
        axes[0].loglog([r['rtol'] for r in output['runs']], [r[key] for r in output['runs']], 'o-', label=label)
    for label, key, color in [('GC: initial moment', 'instantaneous', 'tab:orange'),
                               ('GC: mean moment', 'averaged', 'tab:green')]:
        axes[0].loglog([r['rtol'] for r in output['runs']],
                       [r[key]['gc_max_relative_energy_error'] for r in output['runs']], 'o-', label=label, color=color)
        axes[1].loglog([r['rtol'] for r in output['runs'][:-1]],
                       [abs(r[key]['max_position_error']-output['runs'][-1][key]['max_position_error'])
                        for r in output['runs'][:-1]], 'o-', label=label, color=color)
    axes[0].set(xlabel='Relative solver tolerance', ylabel='Maximum relative\nconserved-energy error')
    axes[1].set(xlabel='Relative solver tolerance', ylabel='Maximum-error change\nfrom tightest run')
    for ax in axes:
        ax.legend(fontsize=10)
    save_figure(fig, 'task4_convergence')
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, d['energy']/d['energy'][0]-1, label='Full-orbit kinetic energy')
    ax.set(xlabel='$t$ (normalized)', ylabel='$K(t)/K(0)-1$')
    ax.legend()
    save_figure(fig, 'task4_energy')

def plot_saved():
    """Redraw poster figures from the saved numerical data without reintegrating."""
    import json
    output = json.loads((ROOT/'results/task4_summary.json').read_text(encoding='utf-8'))
    def read(name):
        return np.genfromtxt(ROOT/'results'/name, names=True, delimiter=',')
    orbit = read('task4_orbit.csv')
    centers = read('task4_guiding_centers.csv')
    correction = read('task4_moment_correction.csv')
    cycles = read('task4_gyrocycle_means.csv')
    def columns(table, names):
        return np.column_stack([table[n] for n in names])
    data = dict(t=orbit['t'], states=columns(orbit, ('x', 'y', 'z', 'vx', 'vy', 'vz')),
                diagnostic=dict(center=columns(orbit, ('X', 'Y', 'Z')), mu=orbit['mu'], energy=orbit['K']),
                instant=columns(centers, ('X_initial_mu', 'Y_initial_mu', 'Z_initial_mu', 'u_initial_mu')),
                averaged=columns(centers, ('X_mean_mu', 'Y_mean_mu', 'Z_mean_mu', 'u_mean_mu')),
                instant_error=centers['error_initial_mu'], averaged_error=centers['error_mean_mu'],
                cycle_time_mu=cycles['time_mean_mu'], cycle_phase_mu=cycles['uniform_phase_mean_mu'],
                cycle_edges_time=np.r_[cycles['cycle_start'], cycles['cycle_end'][-1]],
                eq4_lhs=correction['mu_ratio_minus_one'], eq4_rhs=correction['minus_rho_dot_gradB_over_B'],
                eq4_residual=correction['residual'], curvature_diagnostic=correction['curvature_diagnostic'])
    style()
    make_figures(output, data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true', help='Redraw figures from saved CSV and JSON data')
    args = parser.parse_args()
    if args.plots_only:
        plot_saved()
    else:
        run()
