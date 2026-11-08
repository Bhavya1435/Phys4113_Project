"""Independent geometry, dynamics, averaging and saved-output checks for Task 4."""
import importlib
import json
import unittest
import numpy as np
from scipy.integrate import quad
from common import ROOT, diagnostics

task = importlib.import_module('04_guiding_center')

class Task4Checks(unittest.TestCase):
    def test_vector_field_geometry(self):
        points = np.array([[3., 0., 0.], [2.6, .4, 3.2], [3., -1., -4.5], [0., 0., 1.5]])
        h = 1e-4
        batch = task.geometry(points)
        for i, point in enumerate(points):
            B, b, gradient, curvature = task.geometry(point)
            numeric_J = np.column_stack([(task.magnetic(0., point+h*e)-task.magnetic(0., point-h*e))/(2*h)
                                         for e in np.eye(3)])
            numeric_gradient = np.array([(np.linalg.norm(task.magnetic(0., point+h*e))-
                                           np.linalg.norm(task.magnetic(0., point-h*e)))/(2*h)
                                          for e in np.eye(3)])
            def unit(p):
                value = task.magnetic(0., p)
                return value/np.linalg.norm(value)
            numeric_curvature = (unit(point+h*b)-unit(point-h*b))/(2*h)
            np.testing.assert_allclose(task.field_jacobian(point), numeric_J, atol=2e-11)
            np.testing.assert_allclose(gradient, numeric_gradient, atol=2e-10)
            np.testing.assert_allclose(curvature, numeric_curvature, atol=2e-10)
            self.assertLess(abs(np.dot(b, curvature)), 1e-14)
            for scalar_result, vector_result in zip((B, b, gradient, curvature), batch):
                np.testing.assert_allclose(scalar_result, vector_result[i], atol=1e-15)

    def test_initial_center_and_pitch_components(self):
        for phase in (np.pi/2, -np.pi/2, 0., np.pi):
            state = task.initial_state(phase)
            _, mu, center = diagnostics(state[None, :], task.magnetic)
            B, b, _, _ = task.geometry(state[:3])
            u = np.dot(state[3:], b)
            np.testing.assert_allclose(center[0], task.X0, atol=1e-12)
            self.assertAlmostEqual(u, task.UPAR, places=13)
            self.assertAlmostEqual(np.linalg.norm(state[3:]-u*b), task.VPERP, places=13)
            self.assertAlmostEqual(mu[0], task.M*task.VPERP**2/(2*B), places=13)
        self.assertLess(task.initial_state(np.pi/2)[0], task.R0)
        self.assertGreater(task.initial_state(-np.pi/2)[0], task.R0)

    def test_exact_midplane_gc_drift(self):
        """At u=Z=0 the model has an exactly circular grad-B drift orbit."""
        mu, duration = .13, 40.
        sol = task.integrate_center(mu, duration=duration, initial_u=0.)
        t = np.linspace(0., duration, 101)
        B = task.B0*(1-task.R0**2/(2*task.L**2))
        omega = -mu*task.B0/(task.Q*B*task.L**2)
        exact = np.column_stack((task.R0*np.cos(omega*t), task.R0*np.sin(omega*t),
                                 np.zeros_like(t), np.zeros_like(t)))
        np.testing.assert_allclose(sol.sol(t).T, exact, atol=2e-11, rtol=0.)

    def test_initial_curvature_drift(self):
        B, b, grad, curvature = task.geometry(task.X0)
        expected_gradient = np.array([-task.B0*task.R0/task.L**2, 0., 0.])
        np.testing.assert_allclose(grad, expected_gradient, atol=1e-15)
        np.testing.assert_allclose(curvature, expected_gradient/B, atol=1e-15)
        mu = .13
        rhs = task.center_rhs(np.r_[task.X0, task.UPAR], mu)
        expected_y = (mu*grad[0]+task.M*task.UPAR**2*curvature[0])/(task.Q*B)
        self.assertAlmostEqual(rhs[1], expected_y, places=14)
        self.assertLess(rhs[1], 0.)
        self.assertAlmostEqual(rhs[3], 0., places=14)

    def test_time_and_phase_averages(self):
        """Nonuniform gyrophase provides distinct, independently known averages."""
        t = np.linspace(0., 20*np.pi, 10001)
        phase = t+.15*np.sin(t)
        mu = .13+.002*np.cos(phase)
        primary, alternatives, *_ = task.gyro_averages(t, dict(progress=phase, mu=mu))
        expected = quad(lambda tt: .13+.002*np.cos(tt+.15*np.sin(tt)), 0., t[-1], epsabs=1e-12)[0]/t[-1]
        self.assertAlmostEqual(primary, expected, places=10)
        self.assertAlmostEqual(alternatives['whole_orbit_uniform_phase'], .13, places=10)
        self.assertGreater(abs(primary-.13), 1e-5)

    def test_exact_magnetic_moment_chain_rule(self):
        """An exact Lorentz identity checks parallel-projection effects separately from Eq. (4)."""
        state = np.array([2.6, .4, 3.2, .35, .2, .17])
        B, b, gradient, _ = task.geometry(state[:3])
        v = state[3:]
        u = np.dot(v, b)
        mu = task.M*np.dot(v-u*b, v-u*b)/(2*B)
        derivative_b = (task.field_jacobian(state[:3])-np.outer(b, gradient))/B
        expected = -task.M*u/B*np.dot(v, derivative_b@v)-mu/B*np.dot(v, gradient)
        acceleration = task.Q/task.M*np.cross(v, task.magnetic(0., state[:3]))
        tangent = np.r_[v, acceleration]
        h = 1e-5
        mu_plus = diagnostics((state+h*tangent)[None, :], task.magnetic)[1][0]
        mu_minus = diagnostics((state-h*tangent)[None, :], task.magnetic)[1][0]
        self.assertAlmostEqual((mu_plus-mu_minus)/(2*h), expected, places=9)

    def test_saved_results_and_figures(self):
        summary = json.loads((ROOT/'results/task4_summary.json').read_text())
        tight = summary['runs'][-1]
        for run in summary['runs']:
            np.testing.assert_allclose(run['initial_center'], task.X0, atol=1e-12)
        self.assertLess(tight['full_max_relative_energy_error'], 1e-8)
        self.assertLess(tight['instantaneous']['gc_max_relative_energy_error'], 1e-8)
        self.assertLess(tight['averaged']['gc_max_relative_energy_error'], 1e-8)
        self.assertGreater(tight['error_reduction_factor'], 5.)
        self.assertLess(summary['convergence']['max_full_state_change'], 1e-5)
        self.assertLess(summary['convergence']['averaged_max_error_change'], 1e-5)
        self.assertLess(summary['sampling']['mu_average_change'], 1e-8)
        self.assertGreater(tight['equation4']['correlation'], .99)
        control = summary['parallel_speed_control']['equation4']
        self.assertLess(abs(control['fitted_slope']-1), .05)
        self.assertLess(control['rms_residual'], tight['equation4']['rms_residual'])
        orbit = np.genfromtxt(ROOT/'results/task4_orbit.csv', names=True, delimiter=',')
        centers = np.genfromtxt(ROOT/'results/task4_guiding_centers.csv', names=True, delimiter=',')
        correction = np.genfromtxt(ROOT/'results/task4_moment_correction.csv', names=True, delimiter=',')
        for dataset in (orbit, centers, correction):
            for column in dataset.dtype.names:
                self.assertTrue(np.isfinite(dataset[column]).all())
        state = np.column_stack([orbit[c] for c in ('x', 'y', 'z', 'vx', 'vy', 'vz')])
        energy, mu, center = diagnostics(state, task.magnetic)
        np.testing.assert_allclose(energy, orbit['K'], atol=1e-15)
        np.testing.assert_allclose(mu, orbit['mu'], atol=1e-15)
        np.testing.assert_allclose(center, np.column_stack([orbit[c] for c in ('X', 'Y', 'Z')]), atol=1e-14)
        for name, tag in [('instantaneous', 'initial_mu'), ('averaged', 'mean_mu')]:
            predicted = np.column_stack([centers[f'{c}_{tag}'] for c in ('X', 'Y', 'Z')])
            errors = np.linalg.norm(predicted-center, axis=1)
            np.testing.assert_allclose(errors, centers[f'error_{tag}'], atol=1e-14)
            self.assertAlmostEqual(np.max(errors), tight[name]['max_position_error'], places=12)
            self.assertLessEqual(tight[name]['max_cycle_mean_position_difference'], np.max(errors))
        self.assertAlmostEqual(np.sqrt(np.mean(correction['residual']**2)), tight['equation4']['rms_residual'], places=13)
        for stem in ('full_orbit_and_center', 'center_coordinates', 'position_errors', 'azimuth',
                     'magnetic_moment', 'moment_correction', 'convergence', 'energy'):
            for extension in ('pdf', 'png'):
                path = ROOT/'figures'/f'task4_{stem}.{extension}'
                self.assertGreater(path.stat().st_size, 1000)
                signature = path.read_bytes()[:8]
                self.assertTrue(signature.startswith(b'%PDF') if extension == 'pdf'
                                else signature == b'\x89PNG\r\n\x1a\n')

if __name__ == '__main__':
    unittest.main(verbosity=2)
