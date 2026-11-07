"""Task 1: full Lorentz orbits and fitted guiding-center drift velocities."""
import numpy as np
from common import (ROOT, TOLERANCES, integrate, diagnostics, linear_fit,
                    style, save_figure, save_json, plt)

Q = M = B0 = 1.
VPERP, E0, LG = .5, .05, 100.

def uniform(t, x):
    return np.broadcast_to([0., 0., B0], np.asarray(x).shape)

def electric(t, x):
    return np.broadcast_to([E0, 0., 0.], np.asarray(x).shape)

def gradient(t, x):
    x = np.asarray(x)
    result = np.zeros_like(x)
    result[..., 2] = B0*(1+x[..., 0]/LG)
    return result

def run():
    style()
    # E along +x implies drift along -y. The initial laboratory
    # perpendicular speed is VPERP in both cases.
    duration = 200*2*np.pi*M/(Q*B0)
    t = np.linspace(0, duration, 40001)
    output = {'parameters': dict(q=Q, m=M, B0=B0, vperp=VPERP,
                                electric=E0, Lg=LG, duration=duration), 'cases': {}}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    comparison, ax = plt.subplots(figsize=(9, 5))
    for row, (name, field, efield, initial, theory) in enumerate([
        ('exb', uniform, electric, [0., 0., 0., VPERP, 0., 0.], -E0/B0),
        # x=0 and vy=0 place the nominal guiding center at x=0.
        ('gradb', gradient, None, [0., 0., 0., VPERP, 0., 0.],
         M*VPERP**2/(2*Q*B0*LG))]):
        runs, solutions = [], []
        for rtol, atol in TOLERANCES:
            kwargs = {} if efield is None else {'electric': efield}
            sol = integrate(initial, duration, field, rtol=rtol, atol=atol, **kwargs)
            states = sol.sol(t).T
            energy, mu, center = diagnostics(states, field)
            # Remove the constant drift before computing the gyroradius in E x B.
            if name == 'exb':
                corrected = states.copy()
                corrected[:, 4] -= theory
                center = diagnostics(corrected, field)[2]
            slope, intercept = linear_fit(t, center[:, 1])
            block_slopes = [linear_fit(t[s], center[s, 1])[0]
                            for s in np.array_split(np.arange(len(t)), 8)]
            conserved = energy - Q*E0*states[:, 0] if name == 'exb' else energy
            run = dict(rtol=rtol, atol=atol, drift_y=slope,
                       absolute_theory_error=abs(slope-theory),
                       relative_theory_error=abs((slope-theory)/theory),
                       block_slope_range=float(np.ptp(block_slopes)),
                       max_relative_conserved_energy_error=float(np.max(abs(conserved-conserved[0]))/energy[0]),
                       nfev=sol.nfev)
            runs.append(run)
            solutions.append(states)
        # Change of measured drift with tightened tolerances is an empirical
        # convergence estimate, not a rigorous uncertainty interval.
        result = dict(theory_drift_y=theory, initial_state=initial, runs=runs,
                      tolerance_drift_change=abs(runs[-1]['drift_y']-runs[-2]['drift_y']),
                      max_state_change_last_two=float(np.max(abs(solutions[-1]-solutions[-2]))))
        if name == 'exb':
            # Exact solution for the specified gyrophase and initial position.
            a = E0/B0
            exact = np.column_stack((VPERP*np.sin(t)+a*(1-np.cos(t)),
                                     theory*t+VPERP*(np.cos(t)-1)+a*np.sin(t),
                                     np.zeros_like(t), VPERP*np.cos(t)+a*np.sin(t),
                                     theory-VPERP*np.sin(t)+a*np.cos(t), np.zeros_like(t)))
            result['max_absolute_exact_state_error'] = float(np.max(abs(states-exact)))
        assert result['tolerance_drift_change'] < 1e-8
        assert runs[-1]['relative_theory_error'] < .002
        assert runs[-1]['max_relative_conserved_energy_error'] < 1e-8
        output['cases'][name] = result
        np.savetxt(ROOT/'results'/f'task1_{name}.csv',
                   np.column_stack((t, states, center)), delimiter=',',
                   header='t,x,y,z,vx,vy,vz,X,Y,Z', comments='')
        shown = t <= 8*2*np.pi
        axes[row, 0].plot(states[shown, 0], states[shown, 1], label='Full orbit', lw=1.)
        axes[row, 0].plot(center[shown, 0], center[shown, 1], label='Local guiding center')
        title = r'$E\times B$ drift' if name == 'exb' else r'Gradient-$B$ drift'
        axes[row, 0].set(xlabel='$x$ (normalized)', ylabel='$y$ (normalized)',
                         title=title+' (first 8 gyroperiods)')
        axes[row, 0].set_aspect('equal', adjustable='datalim')
        axes[row, 0].legend(loc='upper right', framealpha=1.)
        axes[row, 1].plot(t, center[:, 1]-intercept, label='Guiding center')
        axes[row, 1].plot(t, theory*t, '--', label='First-order theory')
        axes[row, 1].set(xlabel='$t$ (normalized)', ylabel='$Y(t)-Y_{fit}(0)$ (normalized)')
        axes[row, 1].legend()
        axes[row, 1].text(.05, .1 if name == 'exb' else .7,
                          f'Fitted drift: {slope:+.10f}\nTheory: {theory:+.10f}',
                          transform=axes[row, 1].transAxes, fontsize=12)
        ax.plot(t[shown], (center[:, 1]-intercept-slope*t)[shown], label=title)
    save_figure(fig, 'task1_orbits_and_drifts')
    ax.set(xlabel='$t$ (normalized)', ylabel='Fit residual (normalized)',
           title='Guiding-center residuals: first 8 gyroperiods')
    ax.legend()
    save_figure(comparison, 'task1_fit_residuals')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, case in output['cases'].items():
        axes[0].loglog([r['rtol'] for r in case['runs'][:-1]],
                       [abs(r['drift_y']-case['runs'][-1]['drift_y']) for r in case['runs'][:-1]],
                       'o-', label=name)
        axes[1].loglog([r['rtol'] for r in case['runs']],
                       [r['max_relative_conserved_energy_error'] for r in case['runs']],
                       'o-', label=name)
    axes[0].set(xlabel='Relative solver tolerance', ylabel='Drift change from tightest run')
    axes[1].set(xlabel='Relative solver tolerance', ylabel='Maximum relative energy error')
    for ax in axes:
        ax.legend()
    save_figure(fig, 'task1_convergence')
    save_json('task1_summary.json', output)

if __name__ == '__main__':
    (ROOT/'results').mkdir(exist_ok=True)
    run()
