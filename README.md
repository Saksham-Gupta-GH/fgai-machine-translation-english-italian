# English-to-Italian machine translation — FGAI Assignment 3

**Saksham Gupta · 230911186 · ICT4403**

Compare a Transformer trained from scratch, fine-tuned mT5-small, and prompted LLaMA/Gemma on the same parallel-text test set. Qwen is an additional baseline, not a replacement for either required decoder-only model.

## Current status

- Basic Transformer: trained for 8 epochs; recorded training time 5.17 minutes.
- mT5-small: fine-tuned for 2 epochs; recorded training time 21.48 minutes.
- Qwen 2.5 1.5B: original prompted run completed; corrected evaluation pending.
- LLaMA 3.2 1B-Instruct and Gemma 2 2B-it: evaluation support exists; results are **pending**, not completed experiments.
- Original notebook BLEU/chrF scores used incorrectly shaped references. They are invalid and retained only in `results/legacy_invalid_metrics/`. No winner can be established from those scores.
- Corrected evaluations save full translations, references, dataset hash, sample count, device, parameter count, decoding settings and metrics in `results/corrected/`.

The recovery loader preserves mT5’s separate shared embedding and language-model head (`tie_word_embeddings=False`) and checks the recorded 300,176,768 parameters. This avoids a cross-version loading error from the saved configuration.

The submission PDF is maintained separately and has not been changed by this repair.

## Dataset and experimental setup

Source: [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100), English–Italian.

The executed notebook takes the original training split, strips surrounding whitespace, keeps pairs with 4–35 whitespace-separated words in each language, and retains the first pair per lowercased English source. It collects 32,000 unique sources, shuffles using Python seed 42, then uses 24,000 for training, 3,000 for validation and 1,000 for testing. The remaining 4,000 are unused. Among the 28,000 used pairs the proportions are 85.71%, 10.71%, and 3.57%.

The scratch model uses a 10,000-token SentencePiece BPE vocabulary learned only from the training sentences. mT5 and the decoder-only models use their pretrained tokenizers. Exact-source deduplication reduces overlap; it does not prove the absence of semantic duplicates or pretrained-data contamination.

**Provenance warning:** The original standalone preprocessing script and the executed notebook used different selection/cleaning rules. The recovery workflow reconstructs the notebook dataset and tokenizer from its recorded code. Do not use an unrelated tokenizer with the trained scratch checkpoint. `data/` contains the earlier standalone pipeline data; `recovered_run/data/` is the recovered notebook run used for corrected evaluation.

## Models

| Model | Adaptation | Recorded parameters |
|---|---|---:|
| Basic Transformer | Full training, 4 encoder and 4 decoder layers, width 256, 4 heads | 9,933,824 |
| mT5-small | Full fine-tuning, task prefix `translate English to Italian: ` | 300,176,768 |
| LLaMA 3.2 1B-Instruct | Three-shot prompting; pending execution | Counted at evaluation |
| Gemma 2 2B-it | Three-shot prompting; pending execution | Counted at evaluation |
| Qwen 2.5 1.5B-Instruct | Supplementary three-shot baseline | 1,543,714,304 |

No completed LoRA experiment or FLORES benchmark is claimed. Training history and loss curves are retained. The scratch model's logged `val_ppl` exponentiates label-smoothed loss and is not standard unsmoothed perplexity; it should not be compared directly with mT5 perplexity.

## Run without retraining

```bash
pip install -r requirements.txt
# Recover your downloaded Kaggle archive (weights remain ignored by git).
python scripts/recover_saved_run.py --archive /path/to/results.zip
# Re-evaluate the trained models. No training is performed.
python scripts/evaluate_saved.py --models transformer mt5
# Requires accepted model terms and Hugging Face authentication:
hf auth login
python scripts/evaluate_saved.py --models llama gemma
# Optional additional baseline:
python scripts/evaluate_saved.py --models qwen
```

Model access: sign in and request/accept access on the official [LLaMA model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and [Gemma model page](https://huggingface.co/google/gemma-2-2b-it). Do not commit authentication tokens.

The recovery script needs the source dataset (downloaded or cached). `--cached-arrow /path/to/opus-100-train.arrow` supports offline recovery. Existing recovered data/checkpoints are not overwritten. Keep the original `results.zip` as a backup. Large weights are not uploaded to GitHub.

The evaluation script defaults to the full 1,000-sentence recovered test set, greedy decoding, at most 64 generated tokens and batch size 1 for every model. For faster encoder-decoder checks, use `--batch-size 16 --device cpu --output results/corrected_cpu_batch16` for both models. Those times are amortized milliseconds per sentence (throughput), not individual request latency; keep them separate from batch-size-1 runs.

It warms up each model and synchronizes the accelerator around timing. Models are loaded sequentially. A missing checkpoint/access failure exits visibly instead of claiming a complete four-model comparison. For a smoke test, use `--limit 5 --output /tmp/fgai-smoke`; do not compare that output with full-test results.

To train from scratch intentionally, use `python run_experiments.py`. That is a separate workflow and is not necessary to reuse the saved Kaggle models.

## Evaluation and comparison

Corpus metrics use `sacrebleu.corpus_bleu(predictions, [references])` and equivalent chrF calls. Standard chrF uses `word_order=0`; chrF++ uses `word_order=2`. Prediction/reference counts are checked before scoring. Full outputs are saved so rescoring does not require inference or training.

Compare only runs with matching test hashes, sample counts, hardware and decoding settings. Even with those controls, model size, pretraining, precision, tokenization and adaptation differ: these experiments do not isolate architecture as a causal variable. Old timing values mixed batch sizes (mT5 32, others 1), so the earlier claim of a roughly 30-fold architecture speed advantage is withdrawn.

To rescore completed outputs without loading any models, run `python scripts/summarize_results.py`. Run regression checks with `python -m unittest discover -s tests -v`.

## Verified recovered-model results

Both models below were evaluated on the same 1,000 recovered test sentences, CPU, batch size 16, greedy decoding, with a 64-token generation cap. Training times above are from the original Kaggle run, not this evaluation.

| Model | BLEU | chrF | chrF++ | Mean ms/sentence (batched throughput) |
|---|---:|---:|---:|---:|
| Basic Transformer | 1.61 | 16.26 | 15.53 | 57.79 |
| mT5-small | 12.15 | 34.29 | 32.40 | 221.82 |

mT5 performs better on both quality metrics among these **two recovered models**. This is not a ranking of all four required models. Absolute quality remains limited, and the saved examples contain meaning errors. The original scratch-model BLEU of 30.21 is withdrawn.

[Full comparison CSV](results/corrected_cpu_batch16/quantitative_comparison.csv) · [Transformer predictions](results/corrected_cpu_batch16/transformer.json) · [mT5 predictions](results/corrected_cpu_batch16/mt5.json)

The separate `results/corrected/` folder contains the Transformer CPU batch-size-1 check (also BLEU 1.61). Its timing must not be mixed with this table.

## Temperature analysis

The original Qwen sweep used temperatures 0.2, 0.5, 0.7 and 1.0 on 300 test examples. Its BLEU/chrF scores share the reference-shape error and cannot support a temperature recommendation. Its original diversity calculation also joined sentences before forming n-grams. The corrected notebook uses sentence boundaries for diversity and saves sampled predictions. Run `python scripts/evaluate_saved.py --models qwen --temperature-sweep` to regenerate the baseline and sweep with saved predictions, or rerun the corrected notebook evaluation cell with existing loaded models. No new training is needed.

## Responsible AI: observations supported by saved samples

The 35-probe outputs in `results/ethical_probe_results.json` cover the scratch Transformer, mT5 and Qwen only. They do not establish LLaMA/Gemma behavior.

- Gender: inspect occupation defaults and explicit feminine agreement manually. Keyword-based tags are only screening heuristics, not validated bias measurements.
- Meaning and hallucination: Qwen translates the midnight boundary as `23.00` in one saved probe, changing the time. mT5 also makes substantial lexical errors. Fluency alone is insufficient evidence of fidelity.
- Cultural idioms: saved outputs include failed translations; the earlier broad claim that pretrained models consistently localize idioms is unsupported.
- Sensitive content: the Qwen output for a sentence about a victim begins `L'imputato`, changing the person's role. A `neutral` classifier tag does not establish correctness or safety: non-occupation categories were assigned that tag by default.

Mitigations include bilingual human review, explicit checks of numbers/negation/entities, balanced and counterfactual gender examples, broader cultural test sets, and uncertainty-aware deployment. Automated lexical checks require manual confirmation, especially for grammatical gender and context-dependent meanings.

## Conclusion

The training work is preserved. mT5 leads the recovered two-model comparison; completion of the required comparison remains contingent on actual LLaMA and Gemma evaluation. No missing results are estimated or invented.
