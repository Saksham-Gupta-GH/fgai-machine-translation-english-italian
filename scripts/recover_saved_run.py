"""Recover the original notebook run without retraining neural models."""
import argparse
import json
import os
from pathlib import Path
import zipfile

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('recovered_run'))
    ap.add_argument('--cached-arrow',type=Path)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        for info in archive.infolist():
            name=info.filename
            if name=='basic_transformer_best.pt' or name.startswith('mt5_small_best/'):
                target=(output/'results'/name).resolve()
                if not target.is_relative_to(output/'results'):raise ValueError('Unsafe archive path')
                if info.is_dir():target.mkdir(parents=True,exist_ok=True)
                elif not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(info) as src, target.open('wb') as dst:
                        import shutil
                        shutil.copyfileobj(src,dst)
    if (output/'data/spm_bpe.model').exists() and (output/'data/processed/test.jsonl').exists():
        print('Existing recovery retained.');return
    import random
    import sentencepiece as spm
    from datasets import load_dataset, Dataset
    notebook=json.loads((root/'notebooks/text_translation_assignment3.ipynb').read_text())
    cell=next(''.join(c['source']) for c in notebook['cells'] if 'source_groups = {}' in ''.join(c['source']))
    if args.cached_arrow:
        dataset=Dataset.from_file(str(args.cached_arrow.resolve()))
        loader=lambda *a,**k:dataset
    else:loader=load_dataset
    (output/'data/processed').mkdir(parents=True,exist_ok=True)
    os.chdir(output)
    exec(compile(cell,'original_notebook_preprocessing','exec'),{'load_dataset':loader,'random':random,'json':json,'spm':spm,'SEED':42})
    print('Recovered data/tokenizer; neural model training was not run.')

if __name__=='__main__':main()
