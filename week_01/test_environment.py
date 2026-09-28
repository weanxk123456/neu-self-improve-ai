import unittest
from dataclasses import asdict
import numpy as np
from environment import Scenario, RampMergeEnv
from run import episode, validate

class TaskTests(unittest.TestCase):
    def test_seed_replay_and_action_effect(self):
        env = RampMergeEnv()
        try:
            a,_=env.reset(seed=7)
            env.step(0)
            left=env.vehicle.position.copy()
            b,_=env.reset(seed=7)
            np.testing.assert_array_equal(a,b)
            env.step(1)
            self.assertGreater(abs(left[1]-env.vehicle.position[1]),0.1)
        finally:
            env.close()

    def test_rule_success_and_idle_failure(self):
        self.assertTrue(episode(Scenario(),0)['success'])
        env=RampMergeEnv()
        try:
            env.reset(seed=0)
            for _ in range(60):
                _,_,t,tr,info=env.step(1)
                if t or tr: break
            self.assertFalse(info['success'])
            self.assertTrue(t or tr)
        finally:
            env.close()

    def test_collision_cannot_count_as_success(self):
        env=RampMergeEnv()
        try:
            env.reset(seed=0)
            env.vehicle.position=np.array([env.scenario.merge_length+40,4.])
            env.vehicle.crashed=True
            self.assertFalse(env.success())
            self.assertEqual(env._reward(1),-1)
        finally:
            env.close()

    def test_bad_design_rejected(self):
        for patch in [{'spacing':float('nan')},{'vehicles':2.5},{'merge_length':10}, {'vehicles':True}]:
            d=asdict(Scenario());d.update(patch)
            with self.assertRaises(ValueError): Scenario.from_dict(d)
        validate(Scenario(),[0,1])

if __name__=='__main__': unittest.main()
