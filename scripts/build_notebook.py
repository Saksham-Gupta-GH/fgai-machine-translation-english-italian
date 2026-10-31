"""Export an unexecuted copy of the maintained notebook without duplicating its code."""
import json
from pathlib import Path


def build_self_contained_notebook(output_path='notebooks/text_translation_assignment3_clean.ipynb', probe_json_path=None):
    source=Path(__file__).resolve().parents[1]/'notebooks/text_translation_assignment3.ipynb'
    target=Path(output_path)
    if target.resolve()==source.resolve():
        raise ValueError('Choose a separate output path to preserve the executed training record')
    notebook=json.loads(source.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            cell['outputs']=[]
            cell['execution_count']=None
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
    return str(target)

if __name__=='__main__':
    print(build_self_contained_notebook())
