"""Evaluate saved models without training. Each model writes predictions immediately."""
import argparse
from importlib.metadata import version
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.evaluate import evaluate_translations


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize()
    elif device.type == 'mps': torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='+', choices=['transformer','mt5','llama','gemma','qwen'], default=['transformer','mt5','llama','gemma'])
    ap.add_argument('--run-dir', type=Path, default=Path('recovered_run'))
    ap.add_argument('--output', type=Path, default=Path('results/corrected'))
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--device', choices=['auto','cpu','mps','cuda'], default='auto')
    ap.add_argument('--temperature-sweep', action='store_true')
    ap.add_argument('--sweep-limit', type=int, default=300)
    args = ap.parse_args()
    if args.sweep_limit < 1: ap.error('--sweep-limit must be positive')
    if args.temperature_sweep and any(k not in ('llama','gemma','qwen') for k in args.models):
        ap.error('--temperature-sweep currently supports decoder-only models')
    if args.limit < 1 or args.batch_size < 1: ap.error('--limit and --batch-size must be positive')
    if args.batch_size != 1 and any(k in args.models for k in ('llama','gemma','qwen')):
        ap.error('Decoder-only evaluation currently requires --batch-size 1')
    args.output.mkdir(parents=True, exist_ok=True)
    test_file=args.run_dir/'data/processed/test.jsonl'
    pairs=[json.loads(s) for s in test_file.read_text().splitlines()][:args.limit]
    refs=[p['it'] for p in pairs]
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.device != 'auto': device=torch.device(args.device)
    torch.set_num_threads(4)
    for key in args.models:
        print(f'Evaluating {key}: {len(pairs)} sentences, {device}, batch_size={args.batch_size}', flush=True)
        try:
            if key == 'transformer':
                import sentencepiece as spm
                from src.models import BasicTransformer
                sp=spm.SentencePieceProcessor(model_file=str(args.run_dir/'data/spm_bpe.model'))
                model=BasicTransformer(vocab_size=sp.get_piece_size())
                model.load_state_dict(torch.load(args.run_dir/'results/basic_transformer_best.pt',map_location='cpu',weights_only=True))
                model.to(device).eval()
                def predict_batch(texts):
                    tensors=[torch.tensor([sp.bos_id()]+sp.encode(text)+[sp.eos_id()]) for text in texts]
                    inputs=torch.nn.utils.rnn.pad_sequence(tensors,batch_first=True,padding_value=sp.pad_id()).to(device)
                    outputs=model.translate_greedy(inputs).tolist()
                    return [sp.decode([i for i in out if i not in (sp.bos_id(),sp.eos_id(),sp.pad_id())]) for out in outputs]
            elif key == 'mt5':
                from transformers import PreTrainedTokenizerFast, MT5ForConditionalGeneration
                checkpoint=args.run_dir/'results/mt5_small_best'
                tok=PreTrainedTokenizerFast(tokenizer_file=str(checkpoint/'tokenizer.json'), pad_token='<pad>', eos_token='</s>', unk_token='<unk>', model_input_names=['input_ids','attention_mask'])
                model=MT5ForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.float32,tie_word_embeddings=False).to(device).eval()
                if sum(p.numel() for p in model.parameters()) != 300176768:
                    raise ValueError('mT5-small parameter count differs from the saved training run')
                def predict_batch(texts):
                    inp=tok(['translate English to Italian: '+text for text in texts],padding=True,return_tensors='pt',truncation=True,max_length=64).to(device)
                    out=model.generate(**inp,max_new_tokens=64,do_sample=False,num_beams=1)
                    return [text.strip() for text in tok.batch_decode(out,skip_special_tokens=True)]
            else:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from src.llm_translation import FEW_SHOT_PAIRS
                model_id={'llama':'meta-llama/Llama-3.2-1B-Instruct','gemma':'google/gemma-2-2b-it','qwen':'Qwen/Qwen2.5-1.5B-Instruct'}[key]
                tok=AutoTokenizer.from_pretrained(model_id)
                dtype=torch.float16 if device.type in ('cuda','mps') else torch.float32
                model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=dtype).to(device).eval()
                sampling_temperature=None
                def predict(text):
                    instruction='Translate English to Italian. Output only the translation.'
                    messages=[] if key=='gemma' else [{'role':'system','content':instruction}]
                    for i,(en,it) in enumerate(FEW_SHOT_PAIRS):
                        messages.extend([{'role':'user','content':(instruction+'\n' if key=='gemma' and i==0 else '')+'English: '+en},{'role':'assistant','content':it}])
                    messages.append({'role':'user','content':'English: '+text})
                    prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    inp=tok(prompt,add_special_tokens=False,return_tensors='pt').to(device)
                    out=model.generate(**inp,max_new_tokens=64,do_sample=sampling_temperature is not None,pad_token_id=tok.eos_token_id,**({'temperature':sampling_temperature,'top_p':1.0,'top_k':0} if sampling_temperature is not None else {}))
                    pred=tok.decode(out[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip().split('\n')[0]
                    return pred[8:].strip() if pred.lower().startswith('italian:') else pred
            if key in ('llama','gemma','qwen'):
                predict_batch=lambda texts: [predict(text) for text in texts]
            with torch.inference_mode():
                predict_batch([p['en'] for p in pairs[:args.batch_size]]) # warmup excluded
                sync(device); start=time.perf_counter();predictions=[]
                for i in range(0,len(pairs),args.batch_size):
                    predictions.extend(predict_batch([p['en'] for p in pairs[i:i+args.batch_size]]))
                    if i % (args.batch_size * 10)==0: print(f'{key}: {len(predictions)}/{len(pairs)}',flush=True)
                sync(device);elapsed=time.perf_counter()-start
            record={'runtime':{name:version(name) for name in ['torch','transformers','sacrebleu','sentencepiece']},'model':key,'test_sha256':hashlib.sha256(test_file.read_bytes()).hexdigest(),'sample_count':len(pairs),'device':str(device),'batch_size':args.batch_size,'timing_measure':'mean milliseconds per sentence (amortized throughput for batches)','max_new_tokens':64,'do_sample':False,'total_params':sum(p.numel() for p in model.parameters()),'latency_ms':elapsed/len(pairs)*1000,'references':refs,'source':[p['en'] for p in pairs],'predictions':predictions,**evaluate_translations(predictions,refs)}
            (args.output/f'{key}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
            print({k:record[k] for k in ['model','bleu','chrf','latency_ms']},flush=True)
            if key in ('llama','gemma'):
                probes=json.loads((Path(__file__).resolve().parents[1]/'data/ethical_probes_35.json').read_text())
                with torch.inference_mode():
                    probe_outputs=[dict(item, prediction=predict(item['en']), review_status='manual review required') for item in probes]
                (args.output/f'{key}_ethical_probes.json').write_text(json.dumps(probe_outputs,ensure_ascii=False,indent=2))

            if args.temperature_sweep:
                sweep_pairs=pairs[:args.sweep_limit]
                sweep_dir=args.output/'temperature';sweep_dir.mkdir(exist_ok=True)
                for temperature in [0.2,0.5,0.7,1.0]:
                    sampling_temperature=temperature
                    torch.manual_seed(42)
                    with torch.inference_mode():
                        samples=[predict(pair['en']) for pair in sweep_pairs]
                    gold=[pair['it'] for pair in sweep_pairs]
                    result={'model':key,'temperature':temperature,'seed':42,'sample_count':len(samples),'source':[p['en'] for p in sweep_pairs],'references':gold,'predictions':samples,**evaluate_translations(samples,gold)}
                    (sweep_dir/f'{key}_{temperature}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
                    print(f'{key} temperature {temperature}: BLEU {result["bleu"]}',flush=True)
            del model;gc.collect()
            if device.type=='mps':torch.mps.empty_cache()
            elif device.type=='cuda':torch.cuda.empty_cache()
        except Exception as exc:
            print(f'{key} INCOMPLETE: {type(exc).__name__}: {exc}',flush=True)
            raise

if __name__=='__main__': main()
