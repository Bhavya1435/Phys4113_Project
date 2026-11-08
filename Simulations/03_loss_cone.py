"""Task 3: periodic vacuum magnetic mirror and finite-time loss cone."""
import numpy as np
from scipy.special import i0, i1
from scipy.optimize import brentq
from common import (ROOT, TOLERANCES, integrate, diagnostics, style,
                    save_figure, save_json, plt)

DELTA, K, B0 = .5, .05, 1.
Q = M = 1.
SPEED, DURATION = .5, 900.
Z0 = np.pi/K
HALF_CELL = np.pi/K
MIRROR_RATIO = (1+DELTA)/(1-DELTA)
CRITICAL_ANGLE = float(np.degrees(np.arcsin(1/np.sqrt(MIRROR_RATIO))))
COARSE_ANGLES = [25., 28., 31., 34., 35., 36., 39., 42., 45., 50., 55., 60.]
FINE_ANGLES = np.round(np.arange(35.20, 35.321, .01), 8).tolist()

def mirror_field(t, position):
    """Br e_r + Bz e_z with I1(kr)/r evaluated regularly at r=0.

    I1(s)/s = 1/2+s^2/16+s^4/384+O(s^6), s=kr.
    Thus (Bx,By)=B0*delta*k*[I1(s)/s]*sin(kz)*(x,y).
    """
    x, y, z = np.moveaxis(np.asarray(position, dtype=float), -1, 0)
    s = K*np.hypot(x, y)
    ratio = np.empty_like(s)
    small = abs(s) < 1e-4
    np.divide(i1(s), s, out=ratio, where=~small)
    ratio = np.where(small, .5+s*s/16+s**4/384, ratio)
    factor = B0*DELTA*K*ratio*np.sin(K*z)
    return np.stack((factor*x, factor*y, B0*(1+DELTA*i0(s)*np.cos(K*z))), axis=-1)

def initial_state(angle, convention='particle_axis'):
    alpha = np.radians(angle)
    vp, parallel = SPEED*np.sin(alpha), SPEED*np.cos(alpha)
    if convention == 'particle_axis':
        # The PDF specifies the physical particle on axis at the field minimum.
        return np.array([0., 0., Z0, vp, 0., parallel])
    if convention == 'local_center_axis':
        # Alternative convention for sensitivity only: x=rho, vy=-vperp.
        rho = brentq(lambda r: r*B0*(1-DELTA*i0(K*r))-M*vp/Q, 0., 2.)
        return np.array([rho, 0., Z0, 0., -vp, parallel])
    raise ValueError(convention)

def barrier(t, state):
    return abs(state[2]-Z0)-HALF_CELL
barrier.direction = 1

def turning(t, state):
    return state[5]

def summarize(sol, angle, duration, rtol, atol, convention):
    t = np.linspace(0., duration, int(np.ceil(duration/.1))+1)
    states = sol.sol(t).T
    energy, mu, center = diagnostics(states, mirror_field)
    escaped = sol.t_events[0][sol.t_events[0] <= duration+1e-9]
    selected = sol.t_events[1] <= duration+1e-9
    turn_t = sol.t_events[1][selected]
    turn_z = np.asarray(sol.y_events[1]).reshape(-1, 6)[selected, 2]
    excursion = np.r_[0., abs(states[-1, 2]-Z0), abs(turn_z-Z0)]
    # Two reversals on opposite sides give evidence of a complete bounce.
    bounced = np.any(turn_z > Z0) and np.any(turn_z < Z0)
    status = ('escaped' if len(escaped) else
              'bouncing_observed' if bounced else 'unresolved')
    if np.sin(np.radians(angle))**2 > 1/MIRROR_RATIO:
        cos_turn = ((1-DELTA)/np.sin(np.radians(angle))**2-1)/DELTA
        theory_excursion = float(np.arccos(np.clip(-cos_turn, -1, 1))/K)
    else:
        theory_excursion = None  # No turning point inside the first cell.
    record = dict(angle_deg=float(angle), duration=float(duration), rtol=rtol, atol=atol,
                  convention=convention, classification=status,
                  analytical_trapped=bool(np.sin(np.radians(angle))**2 > 1/MIRROR_RATIO),
                  initial_state=initial_state(angle, convention).tolist(),
                  initial_local_center=center[0].tolist(),
                  max_axial_excursion=float(np.max(excursion)),
                  theory_turning_excursion=theory_excursion,
                  first_barrier_crossing=float(escaped[0]) if len(escaped) else None,
                  reversal_count=int(len(turn_t)), turning_times=turn_t.tolist(),
                  turning_z=turn_z.tolist(),
                  max_relative_energy_error=float(np.max(abs(energy/energy[0]-1))),
                  mu_relative_peak_to_peak=float(np.ptp(mu)/np.mean(mu)), nfev=sol.nfev)
    return record, t, states

def orbit(angle, duration=DURATION, rtol=1e-10, atol=1e-12, convention='particle_axis'):
    sol = integrate(initial_state(angle, convention), duration, mirror_field,
                    rtol=rtol, atol=atol, events=(barrier, turning))
    return summarize(sol, angle, duration, rtol, atol, convention), sol

def refine_extended_boundary(lower, upper, horizon=3600.):
    """Resolve a separate longer-time bracket without relabeling the 900 scan.

    Both bracket endpoints require positive evidence: barrier crossing or
    opposite-side reversals. An unresolved midpoint stops refinement.
    """
    cache = {}
    def measure(angle):
        if angle not in cache:
            (record, t, states), sol = orbit(angle, duration=horizon, rtol=1e-12, atol=1e-14)
            cache[angle] = (record, states)
            print(f"Extended mirror {angle:.8f}: {record['classification']}", flush=True)
        return cache[angle][0]
    if measure(lower)['classification'] != 'escaped':
        raise RuntimeError('Extended-time lower endpoint did not escape')
    # A fine-grid angle can already have a first reflection at t=900 but
    # require longer to complete the second reversal. Check it explicitly
    # rather than carrying the much wider two-reversal 900 bracket forward.
    for candidate in sorted(a for a in FINE_ANGLES if lower < a < upper):
        status = measure(candidate)['classification']
        if status == 'bouncing_observed':
            upper = candidate
            break
        if status == 'escaped':
            lower = candidate
    if measure(upper)['classification'] != 'bouncing_observed':
        raise RuntimeError('Extended-time bracket endpoints lack the required evidence')
    unresolved = None
    while upper-lower > .0001:
        mid = (lower+upper)/2
        row = measure(mid)
        if row['classification'] == 'escaped':
            lower = mid
        elif row['classification'] == 'bouncing_observed':
            upper = mid
        else:
            unresolved = mid
            break
    convergence = []
    for angle in (lower, upper):
        tight, states = cache[angle]
        (loose, _, loose_states), sol = orbit(angle, duration=horizon, rtol=1e-10, atol=1e-12)
        assert tight['classification'] == loose['classification']
        convergence.append(dict(angle_deg=angle, loose=loose, tight=tight,
                                max_state_change=float(np.max(abs(states-loose_states))),
                                excursion_change=abs(tight['max_axial_excursion']-loose['max_axial_excursion']),
                                crossing_time_change=(abs(tight['first_barrier_crossing']-loose['first_barrier_crossing'])
                                                      if tight['first_barrier_crossing'] is not None else None)))
    return dict(duration=horizon, escaped_angle_deg=lower, bouncing_angle_deg=upper,
                width_deg=upper-lower, unresolved_midpoint=unresolved,
                midpoint_shift_from_theory_deg=(lower+upper)/2-CRITICAL_ANGLE,
                records=[cache[a][0] for a in sorted(cache)], convergence=convergence)

def plot_scan(output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax in axes:
        for status, marker, color in [('escaped', 'v', 'tab:red'),
                                      ('bouncing_observed', 'o', 'tab:blue'),
                                      ('unresolved', 's', 'tab:orange')]:
            rows = [r for r in output['scan'] if r['classification'] == status]
            if rows:
                ax.scatter([r['angle_deg'] for r in rows], [r['max_axial_excursion'] for r in rows],
                           marker=marker, color=color, label=status.replace('_', ' '), s=35)
        ax.axvline(CRITICAL_ANGLE, color='black', ls='--', label='Analytical critical angle')
        extended = output['extended_boundary']
        ax.axvspan(extended['escaped_angle_deg'], extended['bouncing_angle_deg'],
                   color='tab:green', alpha=.5)
        ax.axvline((extended['escaped_angle_deg']+extended['bouncing_angle_deg'])/2,
                   color='tab:green', lw=1.3, label='Extended-time boundary')
        ax.axhline(HALF_CELL, color='gray', ls=':', label='First barrier distance')
        ax.set(xlabel=r'Pitch angle $\alpha$ (degrees)', ylabel='Maximum axial excursion\n(normalized)')
    axes[0].legend(fontsize=10)
    axes[1].set(xlim=(35.19, 35.33), ylim=(40, 160), title='Transition detail; scan duration 900')
    axes[1].text(.04, .85, f"Extended bracket (degrees):\n[{extended['escaped_angle_deg']:.7f}, {extended['bouncing_angle_deg']:.7f}]",
                 transform=axes[1].transAxes, fontsize=11)
    save_figure(fig, 'task3_loss_cone_scan')

def plot_convergence(output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for convergence in output['convergence']:
        runs = convergence['runs']
        label = f"{convergence['angle_deg']:.5f} deg"
        axes[0].loglog([r['rtol'] for r in runs], [r['max_relative_energy_error'] for r in runs], 'o-', label=label)
        changes = [abs(r['max_axial_excursion']-runs[-1]['max_axial_excursion']) for r in runs[:-1]]
        axes[1].loglog([r['rtol'] for r in runs[:-1]], changes, 'o-', label=label)
    axes[0].set(xlabel='Relative solver tolerance', ylabel='Maximum relative $K$ error')
    axes[1].set(xlabel='Relative solver tolerance', ylabel='Excursion change\nfrom tightest run')
    for ax in axes:
        ax.legend(fontsize=10)
    save_figure(fig, 'task3_convergence')

def run():
    style()
    (ROOT/'results').mkdir(exist_ok=True)
    angles = sorted(set(COARSE_ANGLES+FINE_ANGLES+[CRITICAL_ANGLE]))
    records, baseline_states = {}, {}
    for angle in angles:
        (record, t, states), sol = orbit(angle)
        records[angle] = record
        baseline_states[angle] = states
        print(f"Mirror {angle:.8f} deg: {record['classification']}, excursion={record['max_axial_excursion']:.6f}", flush=True)
    # A bracket is an operational finite-time boundary; it is not a proof
    # about infinite-time confinement of the full Lorentz orbit.
    lower = max(a for a in angles if records[a]['classification'] == 'escaped')
    upper = min(a for a in angles if a > lower and records[a]['classification'] == 'bouncing_observed')
    refinement = []
    while upper-lower > .0001:
        mid = (lower+upper)/2
        (record, t, states), sol = orbit(mid)
        refinement.append(record)
        records[mid], baseline_states[mid] = record, states
        if record['classification'] == 'escaped':
            lower = mid
        elif record['classification'] == 'bouncing_observed':
            upper = mid
        else:
            # Do not silently equate non-escape with trapped motion.
            break
    output = dict(parameters=dict(q=Q, m=M, B0=B0, delta=DELTA, k=K, speed=SPEED,
                                  duration=DURATION, z_min=Z0, adjacent_barriers=[0., 2*np.pi/K],
                                  coarse_angles=COARSE_ANGLES, fine_angles=FINE_ANGLES),
                  theory=dict(mirror_ratio=MIRROR_RATIO, critical_angle_deg=CRITICAL_ANGLE),
                  classification_rule='escaped: adjacent barrier crossing; bouncing_observed: no crossing and reversals on both sides; otherwise unresolved. All statements are finite-time.',
                  scan=[records[a] for a in sorted(records)],
                  boundary=dict(duration=DURATION,
                                interpretation='Evidence bracket at the main observation horizon, not an exact physical loss-cone boundary.',
                                escaped_angle_deg=lower, bouncing_angle_deg=upper,
                                width_deg=upper-lower,
                                midpoint_shift_from_theory_deg=(lower+upper)/2-CRITICAL_ANGLE),
                  convergence=[], time_sensitivity=[], initialization_sensitivity=[])
    output['extended_boundary'] = refine_extended_boundary(lower, upper)
    # Full tolerance refinement at the reference angles and both sides of the
    # resolved numerical transition; baseline runs are reused.
    representatives = sorted(set([35., 36., lower, upper]))
    figure_states = {}
    for angle in representatives:
        runs = []
        states_by_tol = []
        for rtol, atol in TOLERANCES:
            if rtol == 1e-10:
                record, states = records[angle], baseline_states[angle]
                t = np.linspace(0., DURATION, len(states))
            else:
                (record, t, states), sol = orbit(angle, rtol=rtol, atol=atol)
            runs.append(record)
            states_by_tol.append(states)
        last, previous = runs[-1], runs[-2]
        convergence = dict(angle_deg=angle, runs=runs,
                           max_state_change_last_two=float(np.max(abs(states_by_tol[-1]-states_by_tol[-2]))),
                           excursion_change_last_two=abs(last['max_axial_excursion']-previous['max_axial_excursion']),
                           crossing_time_change_last_two=(abs(last['first_barrier_crossing']-previous['first_barrier_crossing'])
                                                          if last['first_barrier_crossing'] is not None and previous['first_barrier_crossing'] is not None else None),
                           classifications_agree=len(set(r['classification'] for r in runs)) == 1)
        output['convergence'].append(convergence)
        assert last['max_relative_energy_error'] < 1e-8
        assert convergence['classifications_agree']
        if angle in (35., 36.):
            figure_states[angle] = (t, states)
            np.savetxt(ROOT/'results'/f'task3_orbit_{angle:.0f}.csv',
                       np.column_stack((t, states)), delimiter=',',
                       header='t,x,y,z,vx,vy,vz', comments='')
        # Double the duration and inspect nested observation windows.
        (extended, t_ext, states_ext), long_sol = orbit(angle, duration=1800., rtol=1e-12, atol=1e-14)
        windows = [summarize(long_sol, angle, horizon, 1e-12, 1e-14, 'particle_axis')[0]
                   for horizon in (225., 450., 900., 1800.)]
        output['time_sensitivity'].append(dict(angle_deg=angle, windows=windows))
        print(f'Mirror convergence and duration checks: {angle:.8f} deg', flush=True)
    for angle in (35., 36., CRITICAL_ANGLE):
        (record, t, states), sol = orbit(angle, convention='local_center_axis', rtol=1e-12, atol=1e-14)
        output['initialization_sensitivity'].append(record)
    assert records[35.]['classification'] == 'escaped'
    assert records[36.]['classification'] == 'bouncing_observed'
    assert all(r['max_relative_energy_error'] < 1e-7 for r in output['scan'])
    # Plot representative real-space orbits and axial barrier crossings.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig_xy, ax_xy = plt.subplots(figsize=(9, 5))
    for angle, (t, states) in figure_states.items():
        axes[0].plot(t, states[:, 2]-Z0, label=fr'$\alpha={angle:.0f}^\circ$')
        axes[1].plot(np.hypot(states[:, 0], states[:, 1]), states[:, 2]-Z0,
                     lw=.7, label=fr'${angle:.0f}^\circ$')
        ax_xy.plot(states[:, 0], states[:, 1], lw=.6, label=fr'${angle:.0f}^\circ$')
    for ax in axes:
        ax.axhline(HALF_CELL, color='black', ls='--', lw=1)
        ax.axhline(-HALF_CELL, color='black', ls='--', lw=1)
        ax.set(ylabel='$z-z_{min}$ (normalized)')
        ax.legend()
    axes[0].set(xlabel='$t$ (normalized)', title='Adjacent barrier planes: dashed')
    axes[1].set(xlabel='$r$ (normalized)', title='Meridional projection')
    save_figure(fig, 'task3_representative_orbits')
    ax_xy.set(xlabel='$x$ (normalized)', ylabel='$y$ (normalized)', aspect='equal')
    ax_xy.legend(title='Pitch angle')
    save_figure(fig_xy, 'task3_transverse_orbits')
    plot_scan(output)
    plot_convergence(output)
    fig, ax = plt.subplots(figsize=(10, 6))
    for case in output['time_sensitivity']:
        rows = case['windows']
        ax.plot([r['duration'] for r in rows], [r['max_axial_excursion'] for r in rows],
                'o-', label=f"{case['angle_deg']:.5f} deg")
    ax.axhline(HALF_CELL, color='black', ls='--', label='First barrier distance')
    ax.set(xlabel='Observation duration (normalized)', ylabel='Maximum axial excursion (normalized)')
    ax.legend()
    save_figure(fig, 'task3_duration_sensitivity')
    np.savetxt(ROOT/'results'/'task3_scan.csv',
               [[r['angle_deg'], r['duration'], r['max_axial_excursion'], r['max_relative_energy_error'],
                 float(r['classification'] == 'escaped'), float(r['classification'] == 'bouncing_observed'),
                 r['first_barrier_crossing'] if r['first_barrier_crossing'] is not None else np.nan,
                 r['reversal_count']] for r in output['scan']], delimiter=',',
               header='angle_deg,duration,max_axial_excursion,relative_K_error,escaped,bouncing_observed,first_barrier_time,reversal_count', comments='')
    np.savetxt(ROOT/'results'/'task3_convergence.csv',
               [[c['angle_deg'], r['rtol'], r['atol'], r['max_axial_excursion'], r['max_relative_energy_error']]
                for c in output['convergence'] for r in c['runs']], delimiter=',',
               header='angle_deg,rtol,atol,max_axial_excursion,relative_K_error', comments='')
    save_json('task3_summary.json', output)
    save_json('task3_extended_boundary.json', output['extended_boundary'])

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extend-saved', action='store_true',
                        help='Recompute only the long-duration boundary from a saved scan')
    args = parser.parse_args()
    if args.extend_saved:
        style()
        output = json.loads((ROOT/'results'/'task3_summary.json').read_text(encoding='utf-8'))
        output['extended_boundary'] = refine_extended_boundary(output['boundary']['escaped_angle_deg'],
                                                              output['boundary']['bouncing_angle_deg'])
        plot_scan(output)
        save_json('task3_summary.json', output)
        save_json('task3_extended_boundary.json', output['extended_boundary'])
    else:
        run()
