"""Audit downloaded evidence without loading the language model."""
import hashlib
import json
import math
import sys
from pathlib import Path
from test_data import correct, make_data

folder = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'assets/results/full'
r = json.loads((folder/'results.json').read_text())
expected = make_data(r['sample'])
for split in expected:
    path = folder/f'{split}.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['data_sha256'][split], split
    assert [json.loads(line) for line in path.read_text().splitlines()] == expected[split]
for name, model in r['models'].items():
    for split in ('validation','test'):
        rows = model[split]['examples']
        assert [{k:row[k] for k in ('skill','prompt','answer')} for row in rows] == expected[split], (name,split)
        for row in rows:
            assert row['correct'] == correct(row,row['output']), (name,row)
        for skill in ('triage','extract'):
            actual = {'correct':sum(row['correct'] for row in rows if row['skill']==skill),
                      'total':sum(row['skill']==skill for row in rows)}
            assert actual == model[split]['scores'][skill], (name,split,skill)
for skill, evidence in r['training'].items():
    assert evidence['max_gradient_norm'] > 0 and evidence['changed_tensors'] > 0
    assert evidence['trainable_parameters'] > 0 and evidence['reload_ok'] is True
    assert all(math.isfinite(x) and x >= 0 for x in evidence['losses'])
    assert evidence['examples'] == sum(row['skill']==skill for row in expected['train'])
assert r['merge_reload_ok'] is True
presets = ['linear-balanced','linear-skewed','ties']
def rank(name):
    scores = r['models'][name]['validation']['scores']
    acc = [v['correct']/v['total'] for v in scores.values()]
    return min(acc),sum(acc)
assert max(presets,key=rank) == r['selected_by_validation']
print('Verified hashes, generated splits, all scores/responses, training evidence, reload flags and validation selection.')
for name, model in r['models'].items():
    print(name,model['test']['scores'])
