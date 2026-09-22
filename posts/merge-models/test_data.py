"""CPU-only data/scoring tests; no model download or Compute installation needed."""
import ast
import json
import unittest
from pathlib import Path

source = Path(__file__).with_name('train.py')
namespace = {}
exec('import json, random', namespace)
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in ('make_data', 'correct'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
make_data, correct = namespace['make_data'], namespace['correct']


class DataTests(unittest.TestCase):
    def test_disjoint_deterministic_balanced_splits(self):
        for sample in (True, False):
            data = make_data(sample)
            self.assertEqual(data, make_data(sample))
            sets = {k: {r['prompt'] for r in rows} for k, rows in data.items()}
            for k, rows in data.items():
                self.assertEqual(len(sets[k]), len(rows))
                self.assertEqual(sum(r['skill'] == 'triage' for r in rows), len(rows)//2)
                for other in sets:
                    if other != k:
                        self.assertFalse(sets[k] & sets[other])
            for rows in data.values():
                self.assertTrue(all(correct(r, r['answer']) for r in rows))

    def test_exact_queue_not_keyword(self):
        row = {'skill': 'triage', 'answer': 'RUBY'}
        self.assertTrue(correct(row, ' RUBY\n'))
        for output in ('RUBY or JADE', 'The queue is RUBY.', 'ruby', ''):
            self.assertFalse(correct(row, output))

    def test_json_types_keys_and_values(self):
        row = {'skill': 'extract', 'answer': '{"sku":"AX-12345","qty":1}'}
        self.assertTrue(correct(row, '{"qty": 1, "sku": "AX-12345"}'))
        for output in ('{"sku":"AX-12345","qty":"1"}',
                       '{"sku":"AX-12345","qty":true}',
                       '{"sku":"AX-12345","qty":1.0}',
                       '{"sku":"AX-12345","qty":1,"extra":0}',
                       '{"sku":"AX-12345","qty":2}', '[]', 'null', '```json\n{}\n```'):
            self.assertFalse(correct(row, output))


if __name__ == '__main__':
    unittest.main()
