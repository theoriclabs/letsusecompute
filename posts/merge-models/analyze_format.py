"""Diagnostic only: does removing one outer Markdown fence fix extraction?"""
import json
import re
import sys
from pathlib import Path
from test_data import correct

folder = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'assets/results/full'
r = json.loads((folder/'results.json').read_text())
summary = {}
for name, model in r['models'].items():
    rows = [row for row in model['test']['examples'] if row['skill']=='extract']
    relaxed = 0
    for row in rows:
        text = row['output'].strip()
        match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',text,flags=re.DOTALL)
        relaxed += correct(row,match.group(1) if match else text)
    summary[name] = {'strict_correct':sum(row['correct'] for row in rows),
                     'correct_after_removing_one_outer_fence':relaxed,'total':len(rows)}
output = {'purpose':'Diagnostic only. Does not change official scores, recipes or validation selection.',
          'rule':'Strip whitespace; optionally remove exactly one enclosing ``` or ```json fence; apply the unchanged exact JSON-key/value/type scorer.',
          'models':summary}
(folder/'format-diagnostic.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
