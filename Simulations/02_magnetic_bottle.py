"""Task 2: finite-radius full orbit in a vacuum magnetic bottle."""
import numpy as np
from scipy.optimize import brentq
from common import (ROOT, TOLERANCES, bottle_field, integrate, diagnostics,
                    style, save_figure, save_json, plt)

L, VPERP, VPAR, DURATION = 10., .5, .2, 700.

def midplane(t, state):
    return state[2]

def turning(t, state):
    return state[5]

def run():
    style()
    # At z=0, B is vertical. Solve rho=m*vperp/(q*B(rho,0)) so
    # X=x+m(v x B)/(q B^2)=0 exactly for the local first-order map.
    rho = brentq(lambda r: r*(1-r*r/(2*L*L))-VPERP, .49, .52)
    initial = [rho, 0., 0., 0., -VPERP, VPAR]
    t = np.linspace(0., DURATION, 35001)
    omega = VPERP/L
    predicted_half = np.pi/omega
    predicted_amplitude = L*VPAR/VPERP
    runs, solutions = [], []
    for rtol, atol in TOLERANCES:
        sol = integrate(initial, DURATION, bottle_field, rtol=rtol, atol=atol,
                        events=(midplane, turning))
        states = sol.sol(t).T
        energy, mu, center = diagnostics(states, bottle_field)
        crossings = sol.t_events[0]
        crossings = crossings[crossings > 1e-6]
        halves = np.diff(crossings)
        full_periods = crossings[2:]-crossings[:-2]
        extrema = sol.y_events[1][:, 2]
        run = dict(rtol=rtol, atol=atol, nfev=sol.nfev,
                   half_period=float(np.mean(halves)),
                   half_period_min=float(np.min(halves)), half_period_max=float(np.max(halves)),
                   full_period=float(np.mean(full_periods)),
                   zmax=float(np.max(abs(extrema))),
                   turning_amplitude_range=float(np.ptp(abs(extrema))),
                   mu_initial=float(mu[0]), mu_mean=float(np.mean(mu)),
                   mu_absolute_peak_to_peak=float(np.ptp(mu)),
                   mu_relative_peak_to_peak=float(np.ptp(mu)/np.mean(mu)),
                   mu_max_relative_change_from_initial=float(np.max(abs(mu/mu[0]-1))),
                   energy_max_relative_error=float(np.max(abs(energy/energy[0]-1))),
                   midplane_crossings=crossings.tolist(),
                   turning_times=sol.t_events[1].tolist(), turning_z=extrema.tolist())
        runs.append(run)
        solutions.append(states)
    tight = runs[-1]
    # Independently refine the output grid for the sampled magnetic moment
    # extrema. solve_ivp internal steps are adaptive and unrelated to this grid.
    refined_mu = diagnostics(sol.sol(np.linspace(0., DURATION, 70001)).T, bottle_field)[1]
    refined_mu_range = float(np.ptp(refined_mu)/np.mean(refined_mu))
    output = dict(parameters=dict(L=L, B0=1., q=1., m=1., vperp=VPERP, vpar=VPAR,
                                  duration=DURATION, initial_state=initial),
                  theory=dict(half_period=predicted_half, full_period=2*predicted_half,
                              zmax=predicted_amplitude, omega=omega), runs=runs,
                  convergence=dict(max_state_change_last_two=float(np.max(abs(solutions[-1]-solutions[-2]))),
                                   half_period_change=abs(tight['half_period']-runs[-2]['half_period']),
                                   zmax_change=abs(tight['zmax']-runs[-2]['zmax']),
                                   mu_range_change=abs(tight['mu_relative_peak_to_peak']-runs[-2]['mu_relative_peak_to_peak'])),
                  sampling=dict(refined_mu_relative_peak_to_peak=refined_mu_range,
                                mu_range_change=abs(refined_mu_range-tight['mu_relative_peak_to_peak']),
                                output_dt=.02, refined_output_dt=.01),
                  theory_errors={key: dict(absolute=abs(tight[key]-prediction),
                                          relative=abs(tight[key]/prediction-1))
                                 for key, prediction in [('half_period', predicted_half), ('zmax', predicted_amplitude)]})
    assert tight['energy_max_relative_error'] < 1e-8
    assert abs(tight['half_period']/predicted_half-1) < .01
    assert abs(tight['zmax']/predicted_amplitude-1) < .01
    assert tight['mu_relative_peak_to_peak'] < .005
    assert output['convergence']['half_period_change'] < 1e-6
    assert output['convergence']['zmax_change'] < 1e-6
    np.savetxt(ROOT/'results'/'task2_orbit.csv',
               np.column_stack((t, states, mu, energy)), delimiter=',',
               header='t,x,y,z,vx,vy,vz,mu,kinetic_energy', comments='')
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, states[:, 2], label='Full orbit')
    axes[0].plot(t, predicted_amplitude*np.sin(omega*t), '--', label='Guiding-center theory')
    axes[0].set(ylabel='$z$ (normalized)')
    axes[0].legend()
    axes[1].plot(t, states[:, 2]-predicted_amplitude*np.sin(omega*t))
    axes[1].set(xlabel='$t$ (normalized)', ylabel='$z-z_{GC}$ (normalized)')
    save_figure(fig, 'task2_bounce')
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, 1e4*(mu/mu[0]-1), lw=.8)
    axes[0].set(ylabel='$10^4[\\mu(t)/\\mu(0)-1]$')
    axes[1].plot(t, energy/energy[0]-1, lw=.8)
    axes[1].set(xlabel='$t$ (normalized)', ylabel='$K(t)/K(0)-1$')
    save_figure(fig, 'task2_invariants')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    ax.plot(states[::5, 0], states[::5, 1], states[::5, 2], lw=.6)
    ax.set(xlabel='$x$ (normalized)', ylabel='$y$ (normalized)', zlabel='$z$ (normalized)')
    save_figure(fig, 'task2_orbit')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, label in zip(axes, ('half_period', 'zmax'),
                              (r'$\Delta(T_b/2)$ (normalized)', r'$\Delta\max|z|$ (normalized)')):
        ax.loglog([r['rtol'] for r in runs[:-1]],
                  [abs(r[key]-tight[key]) for r in runs[:-1]], 'o-')
        ax.set(xlabel='Relative solver tolerance', ylabel=label,
               title='Difference from tightest run')
    save_figure(fig, 'task2_convergence')
    save_json('task2_summary.json', output)

if __name__ == '__main__':
    (ROOT/'results').mkdir(exist_ok=True)
    run()
