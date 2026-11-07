"""Independent sign/geometry checks plus validation of saved scientific outputs.

Run the two task scripts first. No network access or course PDF is needed.
"""
import json
import unittest
import numpy as np
from common import ROOT, integrate, bottle_field


class PhysicsChecks(unittest.TestCase):
    def test_uniform_orbit_both_charge_signs(self):
        """Exact cyclotron solution checks handedness and q/m dependence."""
        for q in (-1., 1.):
            def field(t, x):
                return np.array([0., 0., 1.])
            t = np.linspace(0., 20., 401)
            states = integrate([0., 0., 0., .5, 0., .2], 20., field, q=q).sol(t).T
            exact = np.column_stack((.5*np.sin(q*t)/q,
                                     .5*(np.cos(q*t)-1)/q, .2*t,
                                     .5*np.cos(q*t), -.5*np.sin(q*t),
                                     np.full_like(t, .2)))
            np.testing.assert_allclose(states, exact, atol=2e-11, rtol=0.)

    def test_bottle_vacuum_geometry(self):
        """Finite differences independently verify div B and curl B."""
        for point in (np.array([.5, -.3, 4.]), np.array([0., 0., 0.])):
            h = 1e-4
            jac = np.column_stack([(bottle_field(0., point+h*e)-
                                    bottle_field(0., point-h*e))/(2*h)
                                   for e in np.eye(3)])
            self.assertLess(abs(np.trace(jac)), 1e-10)
            np.testing.assert_allclose(jac, jac.T, atol=1e-10)

    def test_saved_results_and_figures(self):
        drift = json.loads((ROOT/'results'/'task1_summary.json').read_text())
        bottle = json.loads((ROOT/'results'/'task2_summary.json').read_text())
        self.assertLess(drift['cases']['exb']['max_absolute_exact_state_error'], 1e-8)
        for name, sign in [('exb', -1), ('gradb', 1)]:
            case = drift['cases'][name]
            self.assertGreater(sign*case['runs'][-1]['drift_y'], 0.)
            self.assertLess(case['tolerance_drift_change'], 1e-8)
            self.assertLess(case['runs'][-1]['relative_theory_error'], .002)
        self.assertLess(bottle['runs'][-1]['energy_max_relative_error'], 1e-8)
        self.assertLess(bottle['convergence']['half_period_change'], 1e-6)
        self.assertLess(bottle['convergence']['zmax_change'], 1e-6)
        # Check the on-axis initial guiding center from the chosen local map.
        state = np.array(bottle['parameters']['initial_state'])
        B = bottle_field(0., state[:3])
        X = state[:3]+np.cross(state[3:], B)/np.dot(B, B)
        np.testing.assert_allclose(X, 0., atol=1e-12)
        for name in ('task1_orbits_and_drifts', 'task1_fit_residuals', 'task1_convergence',
                     'task2_bounce', 'task2_invariants', 'task2_orbit', 'task2_convergence'):
            for extension in ('pdf', 'png'):
                path = ROOT/'figures'/f'{name}.{extension}'
                self.assertGreater(path.stat().st_size, 1000)
                prefix = path.read_bytes()[:8]
                self.assertTrue(prefix.startswith(b'%PDF') if extension == 'pdf'
                                else prefix == b'\x89PNG\r\n\x1a\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
