"""Rescore stored predictions and produce a comparison table without inference."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluate import evaluate_translations

folder=Path(sys.argv[1] if len(sys.argv)>1 else 'results/corrected')
rows=[]
protocol=None
for path in sorted(folder.glob('*.json')):
    record=json.loads(path.read_text())
    if not isinstance(record,dict) or 'predictions' not in record:continue
    current=tuple(record[k] for k in ['test_sha256','sample_count','device','batch_size','max_new_tokens','do_sample'])
    if protocol is not None and current != protocol:
        raise ValueError(f'Incompatible evaluation protocols: {path}')
    protocol=current
    metrics=evaluate_translations(record['predictions'],record['references'])
    rows.append({**{k:record[k] for k in ['model','sample_count','total_params','device','batch_size','latency_ms']},**metrics})
if not rows:raise SystemExit('No completed evaluation files found')
with (folder/'quantitative_comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(rows,indent=2))
