"""Check DP against an independent closed-form solution for this small MDP."""
import tempfile
import unittest
from mdp import run

class MergeMDPTests(unittest.TestCase):
    def test_policy_iteration_and_closed_form(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory)
            again = run(directory)
        self.assertEqual(result, again)
        final = result['history'][-1]
        self.assertEqual(final['policy'], {'0,0': 0., '0,1': 1., '1,0': 1., '1,1': 0.})
        # V(safe)=.98-.02; V(unsafe)=-.0595+.95*(.30*.96+.65*V(unsafe)).
        self.assertAlmostEqual(final['values'][0], .96, places=8)
        self.assertAlmostEqual(final['values'][1], (-.0595+.95*.30*.96)/(1-.95*.65), places=8)
        self.assertEqual(final['values'][2], 0.)
        self.assertEqual(final['values'][3], 0.)
        self.assertLess(result['bellman_optimality_residual'], 1e-8)
        for old, new in zip(result['history'], result['history'][1:]):
            for state in [0, 1]:
                self.assertGreaterEqual(new['values'][state]+1e-8, old['values'][state])

if __name__ == '__main__':
    unittest.main()
