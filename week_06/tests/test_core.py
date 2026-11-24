import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from core import Agent, DIRECT, GENERATOR, PLANNER, REASONER, VERIFIER, advantages, exact_match, load_data, parse_json
from tools import Tools


class FakeModel:
    def __init__(self, finish=False, malformed=False):
        self.calls = []
        self.finish, self.malformed = finish, malformed

    def generate(self, system, user, adapter=False, temperature=0):
        self.calls.append((system, adapter, temperature))
        if system == PLANNER:
            raw = 'bad JSON' if self.malformed else json.dumps({'tool': 'finish'} if self.finish else
                                                              {'tool': 'search', 'query': 'test query'})
            return raw, [1, 2], [3, 4]
        if system == VERIFIER:
            return '{"done":true,"feedback":"supported"}', [1], [2]
        return 'Rome', [1], [2]


class FakeTools:
    def execute(self, action):
        return 'Rome is the capital of Italy.'


class CoreTests(unittest.TestCase):
    def test_reward_aliases(self):
        self.assertEqual(exact_match('The U.S.A.', ['USA']), 1)
        self.assertEqual(exact_match('Rome or Paris', ['Rome']), 0)

    def test_group_advantage(self):
        self.assertEqual(advantages([0, 0, 0]), [0, 0, 0])
        a = advantages([0, 1, 1, 0])
        self.assertAlmostEqual(sum(a), 0)
        self.assertAlmostEqual(a[0], -1)
        self.assertAlmostEqual(a[1], 1)

    def test_json(self):
        self.assertEqual(parse_json('```json\n{"tool":"finish"}\n```')['tool'], 'finish')
        with self.assertRaises(ValueError):
            parse_json('not JSON')

    def test_only_planner_uses_adapter(self):
        m = FakeModel()
        t = Agent(m, FakeTools()).run('Capital of Italy?', planner_adapter=True, temperature=.5)
        self.assertEqual(t.answer, 'Rome')
        self.assertEqual(len(t.decisions), 1)
        self.assertEqual(t.decisions[0].prompt_ids, [1, 2])
        self.assertTrue(all(adapter == (system == PLANNER) for system, adapter, _ in m.calls))
        self.assertEqual(m.calls[-1][0], GENERATOR)

    def test_finish_and_malformed(self):
        m = FakeModel(finish=True)
        t = Agent(m, FakeTools()).run('Q')
        self.assertEqual(len(m.calls), 2)
        m = FakeModel(malformed=True)
        t = Agent(m, FakeTools()).run('Q')
        self.assertEqual(t.decisions[0].action['tool'], 'invalid')

    def test_real_tool_cache_and_failure_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = Tools(cache_dir=tmp)
            with patch.object(tools, 'wiki', return_value=[{'text': 'evidence'}]) as lookup:
                self.assertEqual(tools.execute({'tool': 'wiki', 'query': 'x'}), '[{"text": "evidence"}]')
                tools.execute({'tool': 'wiki', 'query': 'x'})
                self.assertEqual(lookup.call_count, 1)
            with patch.object(tools, 'wiki', side_effect=RuntimeError('offline')) as lookup:
                for _ in range(2):
                    self.assertIn('ToolError', tools.execute({'tool': 'wiki', 'query': 'y'}))
                self.assertEqual(lookup.call_count, 2)

    def test_data_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.jsonl'
            path.write_text('{"question":"Q","answers":["A"]}\n')
            self.assertEqual(load_data(path)[0]['answers'], ['A'])
            path.write_text('{"question":"Q","answers":[]}\n')
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == '__main__':
    unittest.main()
