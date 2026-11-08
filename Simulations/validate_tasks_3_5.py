"""Independent physical checks for Tasks 3 and 5; run task scripts first."""
import importlib
import json
import unittest
import numpy as np
from common import ROOT

mirror = importlib.import_module('03_loss_cone')
penning = importlib.import_module('05_penning_trap_bhavya')

class PhysicsChecks(unittest.TestCase):
    def test_mirror_axis_and_vacuum_geometry(self):
        points = np.array([[0., 0., 0.], [0., 0., mirror.Z0], [0., 0., 21.]])
        values = mirror.mirror_field(0., points)
        np.testing.assert_allclose(values[:, :2], 0., atol=1e-15)
        self.assertAlmostEqual(values[0, 2]/values[1, 2], 3.)
        for point in (points[2], np.array([1.3, -.7, 43.])):
            h = 1e-4
            jac = np.column_stack([(mirror.mirror_field(0., point+h*e)-
                                    mirror.mirror_field(0., point-h*e))/(2*h)
                                   for e in np.eye(3)])
            self.assertLess(abs(np.trace(jac)), 1e-9)
            np.testing.assert_allclose(jac, jac.T, atol=1e-9)
        # Cartesian regularity at radii approaching zero.
        z = 21.
        expected = mirror.B0*mirror.DELTA*mirror.K*.5*np.sin(mirror.K*z)
        for r in (1e-9, 1e-5):
            self.assertAlmostEqual(mirror.mirror_field(0., [r, 0., z])[0]/r, expected, places=12)

    def test_mirror_pitch_and_initialization(self):
        for angle in (25., 35., 36., 60.):
            for convention in ('particle_axis', 'local_center_axis'):
                state = mirror.initial_state(angle, convention)
                B = mirror.mirror_field(0., state[:3])
                pitch = np.degrees(np.arccos(np.dot(state[3:], B)/(np.linalg.norm(state[3:])*np.linalg.norm(B))))
                self.assertAlmostEqual(pitch, angle, places=10)
                self.assertAlmostEqual(np.linalg.norm(state[3:]), mirror.SPEED, places=12)
                if convention == 'local_center_axis':
                    center = state[:3]+np.cross(state[3:], B)/np.dot(B, B)
                    np.testing.assert_allclose(center, [0., 0., mirror.Z0], atol=1e-12)

    def test_non_escape_is_not_trapping(self):
        (record, _, _), _ = mirror.orbit(25., duration=1.)
        self.assertEqual(record['classification'], 'unresolved')
        self.assertEqual(record['reversal_count'], 0)
        self.assertIsNone(record['first_barrier_crossing'])

    def test_electric_field_from_potential(self):
        point = np.array([.8, -.4, .3])
        h = 1e-5
        for q, m in ((1., 1.), (-2., 3.)):
            derivative = np.array([(penning.potential(point+h*e, .4, q=q, m=m)-
                                    penning.potential(point-h*e, .4, q=q, m=m))/(2*h)
                                   for e in np.eye(3)])
            np.testing.assert_allclose(penning.electric_field(.4, q=q, m=m)(0., point),
                                       -derivative, atol=1e-11)

    def test_penning_stability_and_mode_identity(self):
        for ratio in penning.RATIOS:
            ev = np.linalg.eigvals(penning.system_matrix(ratio))
            self.assertLess(np.max(abs(ev.real)), 1e-12)
            frequencies = ev.imag[ev.imag > 0.]
            self.assertAlmostEqual(np.sum(frequencies**2), 1., places=12)
        self.assertGreater(np.max(np.linalg.eigvals(penning.system_matrix(.8)).real), .26)

    def test_saved_scientific_outputs(self):
        m = json.loads((ROOT/'results'/'task3_summary.json').read_text())
        p = json.loads((ROOT/'results'/'task5_bhavya_summary.json').read_text())
        by_angle = {r['angle_deg']: r for r in m['scan']}
        self.assertEqual(by_angle[35.]['classification'], 'escaped')
        self.assertEqual(by_angle[36.]['classification'], 'bouncing_observed')
        self.assertLess(m['extended_boundary']['width_deg'], .00011)
        self.assertAlmostEqual(m['theory']['critical_angle_deg'], 35.264389682754654, places=10)
        for row in m['scan']:
            self.assertLess(row['max_relative_energy_error'], 1e-7)
        for c in m['convergence']:
            self.assertTrue(c['classifications_agree'])
            self.assertLess(c['runs'][-1]['max_relative_energy_error'], 1e-8)
        for c in m['extended_boundary']['convergence']:
            self.assertEqual(c['loose']['classification'], c['tight']['classification'])
            self.assertLess(c['tight']['max_relative_energy_error'], 1e-8)
        for row in m['scan']:
            if row['classification'] == 'bouncing_observed':
                self.assertGreaterEqual(row['reversal_count'], 2)
        for c in p['cases']:
            self.assertLess(c['runs'][-1]['max_relative_frequency_error'], 1e-8)
            self.assertLess(c['runs'][-1]['max_scaled_H_error'], 1e-8)
            self.assertLess(c['matrix_exponential_max_state_error'], 1e-7)
            self.assertLess(abs(c['runs'][-1]['frequency_identity_residual']), 1e-8)
            self.assertTrue(all(s < 0. for s in c['runs'][-1]['signed_phase_slopes']))
        self.assertEqual(p['marginal_stability']['radial_eigenspace_dimension'], 1)
        self.assertGreater(p['marginal_stability']['propagator_norm_t100'],
                           5*p['marginal_stability']['propagator_norm_t10'])
        for task, stems in [(3, ('representative_orbits', 'transverse_orbits', 'loss_cone_scan',
                                 'convergence', 'duration_sensitivity')),
                            (5, ('trajectories', 'radial_modes', 'frequencies_and_drift_error',
                                 'conservation_convergence', 'frequency_convergence'))]:
            for stem in stems:
                for extension in ('pdf', 'png'):
                    prefix = 'task5_bhavya' if task == 5 else 'task3'
                    path = ROOT/'figures'/f'{prefix}_{stem}.{extension}'
                    self.assertGreater(path.stat().st_size, 1000)
                    signature = path.read_bytes()[:8]
                    self.assertTrue(signature.startswith(b'%PDF') if extension == 'pdf'
                                    else signature == b'\x89PNG\r\n\x1a\n')

if __name__ == '__main__':
    unittest.main(verbosity=2)
