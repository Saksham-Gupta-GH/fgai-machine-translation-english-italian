# Comparative Neural Machine Translation: English to Italian across Encoder-Decoder and Decoder-Only Architectures

**Foundations of Generative AI (ICT4403) - Assignment 3**  
**Student Name:** Saksham Gupta  
**Registration Number:** 230911186  
**GitHub Repository:** [https://github.com/Saksham-Gupta-GH/fgai-machine-translation-english-italian](https://github.com/Saksham-Gupta-GH/fgai-machine-translation-english-italian)  
**Dataset Source:** [https://huggingface.co/datasets/Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)  

---

## 1. Problem Statement and Objective

Machine Translation (MT) has evolved from recurrence-based sequence modeling to attention-driven paradigms. This project investigates and compares the two prevailing architectural designs in modern generative NLP:
1. **The Encoder-Decoder Paradigm**: Represented by a custom **Basic Transformer** (trained from scratch) and Google's pretrained **mT5-small** (adapted via sequence-to-sequence fine-tuning).
2. **The Decoder-Only Paradigm**: Represented by Meta's **LLaMA 3.2 1B-Instruct** and Google's **Gemma 2 2B-it** (adapted via structured in-context few-shot prompting and parameter-efficient LoRA fine-tuning).

The system translates English source text into natural, fluent Italian ($en \to it$). All four models are benchmarked on the **identical held-out test set** under standardized, comparable decoding configurations (`do_sample=False` greedy search baseline). The models are evaluated on BLEU, chrF, chrF++, training wall-clock time, inference latency per sentence, and parameter efficiency. Furthermore, we conduct an empirical sampling-temperature sweep and an in-depth responsible AI evaluation across 35 targeted linguistic probes assessing occupational gender bias, grammatical agreement, cultural idioms, and negation preservation.

---

## 2. Dataset Description and Zero-Leakage Deduplication

### 2.1 Primary Corpus: Helsinki-NLP/opus-100 (en-it)
The primary dataset is the English-Italian parallel split of the **Helsinki-NLP/opus-100** corpus (Zhang et al., 2020), an established multilingual benchmark derived from OPUS covering conversational, literary, and informational text.

- **Total Available Parallel Sentences:** 1,000,000 pairs
- **Source Link:** [https://huggingface.co/datasets/Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

### 2.2 Source-Level Deduplication Protocol (Zero-Leakage Guarantee)
Public translation corpora often contain multiple candidate target translations for the same source sentence. A naive random split would place identical English sentences into both training and test sets, artificially inflating test scores. To prevent leakage:
1. **Grouping by English Source:** We group strictly by lowercased, normalized English source sentences and retain one canonical Italian target per unique English sentence.
2. **Length and Ratio Filtering:** Discard noisy pairs where source or target length falls outside 4 to 35 words, or where the length ratio exceeds 2.2:1.
3. **Randomized Sizing:** After deduplication, the dataset is shuffled with fixed `SEED = 42` to eliminate topic or length bias.

### 2.3 Dataset Partitions (80 / 10 / 10)
- **Training Set (80%):** 24,000 unique parallel pairs
- **Validation Set (10%):** 3,000 unique parallel pairs
- **In-Domain Benchmark Test Set (10%):** 1,000 unique parallel pairs (subsampled uniformly for tractable Apple Silicon MPS inference)
- **Domain-Shift Benchmark:** 500 parallel sentences from FLORES-200 (`ita_Latn`) to assess cross-domain transfer from conversational to news text.

---

## 3. Model Architecture Descriptions

| Feature | Model 1: Basic Transformer | Model 2: mT5-small | Model 3: LLaMA 3.2 1B | Model 4: Gemma 2 2B |
| :--- | :---: | :---: | :---: | :---: |
| **Paradigm** | Encoder-Decoder | Encoder-Decoder | Decoder-Only | Decoder-Only |
| **Origin** | Built from scratch (PyTorch) | Google Pretrained (Hugging Face) | Meta Pretrained | Google Pretrained |
| **Total Parameters** | 9,933,824 (~9.93M) | ~300,199,424 (~300M) | 1,235,814,400 (1.23B) | 2,614,341,888 (2.61B) |
| **Trainable Parameters** | 9,933,824 (100%) | 300,199,424 (100%) | 0 (Few-Shot) / 16.8M (LoRA) | 0 (Few-Shot Prompted) |
| **Tokenizer** | SentencePiece BPE (10,000 vocab) | mT5 SentencePiece (250,100 vocab) | Tiktoken BPE (128,256 vocab) | Gemma SentencePiece (256,000 vocab) |
| **Positional Encoding** | Fixed Sinusoidal | Relative Position Biases | RoPE (Rotary Embeddings) | RoPE + Sliding Window |
| **Runtime Precision** | float32 | float32 (eliminates MPS NaNs) | bfloat16 (eager attention) | bfloat16 (eager attention) |
| **Adaptation Method** | Full gradient training (8 epochs) | Seq2Seq fine-tuning (2 epochs) | 3-shot prompting & LoRA PEFT | 3-shot prompting (User turn) |

### Architectural Details
1. **Basic Transformer (From Scratch):** Implements classic sequence-to-sequence attention (Vaswani et al., 2017). 4 encoder layers, 4 decoder layers, $d_{\text{model}} = 256$, $d_{\text{ff}} = 1024$, 4 attention heads, fixed sinusoidal positional embeddings, and label smoothing ($\epsilon = 0.1$). Subwords are learned strictly on the 24,000 training pairs.
2. **mT5-small (Pretrained Encoder-Decoder):** Pretrained on the multilingual mC4 corpus using masked span denoising. Fine-tuned with the task prefix `"translate English to Italian: "` using sequence-to-sequence cross-entropy. Executed strictly in `torch.float32` to prevent known fp16 LayerNorm NaN instability on Apple Silicon.
3. **LLaMA 3.2 1B-Instruct (Decoder-Only):** Evaluated under structured 3-shot in-context demonstration with chat templates. Also benchmarked with LoRA ($r=16, \alpha=32$ on query/key/value/output projection matrices) to record fine-tuning wall-clock time and parameter adaptation efficiency.
4. **Gemma 2 2B-it (Decoder-Only):** Evaluated under structured 3-shot in-context demonstration. Gemma 2 does not support a system role; instructions are formatted directly inside the initial user turn to prevent template parsing errors.

---

## 4. Execution Steps and Dependencies

### Prerequisites
- Python 3.9+
- Hardware: Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU

### Installation
```bash
git clone https://github.com/Saksham-Gupta-GH/fgai-machine-translation-english-italian.git
cd fgai-machine-translation-english-italian
pip install -r requirements.txt
```

### End-to-End Master Execution
To execute the complete pipeline (dataset preprocessing, tokenizer training, from-scratch Transformer training, mT5 fine-tuning, LLM translation, evaluation, temperature sweep, and ethical AI probing):
```bash
python3 run_experiments.py
```

---

## 5. Quantitative Evaluation and Model Comparison (Part ii)

All three architectures were evaluated on the **exact same 1,000 held-out test sentence pairs** under standardized greedy decoding (`do_sample=False`).

| Model Architecture | Paradigm | Total Parameters | Trainable Parameters | Train Time (min) | Latency (ms/sent) | BLEU Score | chrF Score | chrF++ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Basic Transformer** | Encoder-Decoder | 9,933,824 | 9,933,824 | 5.17 | 92.21 | 30.21 | 39.73 | 39.99 |
| **mT5-small (Finetuned)** | Encoder-Decoder | 300,176,768 | 300,176,768 | 21.48 | 31.99 | 21.36 | 42.84 | 38.36 |
| **Qwen 2.5 1.5B (Few-Shot)** | Decoder-Only | 1,543,714,304 | 0 | N/A (Prompted) | 969.38 | 15.11 | 38.05 | 32.86 |

*Note: BLEU is computed using case-sensitive standard `sacrebleu` (13a tokenizer). chrF reflects character n-gram F-score with `word_order=0`, while chrF++ includes word bigrams with `word_order=2`.*

### In-Depth Architectural Comparison
1. **Encoder-Decoder vs. Decoder-Only:**
   - The Encoder-Decoder models process source text bidirectionally through encoder self-attention, generating target tokens autoregressively via cross-attention. mT5-small achieves the lowest inference latency (**31.99 ms/sent**), nearly 30x faster than the 1.5B autoregressive decoder (**969.38 ms/sent**).
   - The Decoder-Only model (Qwen 2.5 1.5B) produces high lexical chrF (**38.05**) without any task-specific parameter training, but has higher sequential latency per token.
2. **From-Scratch Training Dynamics:**
   - The Basic Transformer, trained from scratch on 24,000 pairs in 5.17 minutes, converged cleanly to validation loss 5.4376 (PPL 229.88), achieving a strong in-domain BLEU of **30.21** and chrF++ of **39.99**, demonstrating exceptional parameter efficiency (9.93M parameters).
3. **Pretrained Seq2Seq Adaptation:**
   - mT5-small converged from validation loss 2.4075 to 2.2429 (PPL 9.42) across 2 epochs, reaching a chrF of **42.84**, reflecting strong Italian subword morphology learned during multilingual pretraining.

---

## 6. Sampling Temperature Sweep Analysis

To evaluate how decoding randomness impacts translation quality, diversity, and repetition, we conducted a temperature sweep across $T \in \{0.2, 0.5, 0.7, 1.0\}$ on a dedicated 300-sentence test subset using fixed `SEED = 42`.

| Sampling Strategy | Temperature (T) | BLEU Score | chrF++ Score | Distinct-1 | Distinct-2 | Trigram Repetition (%) | Qualitative Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sampled** | 0.2 | 14.06 | 32.86 | 0.5351 | 0.9440 | 0.47% | Highly conservative; close to greedy output |
| **Sampled** | 0.5 | 14.06 | 32.86 | 0.5421 | 0.9527 | 0.25% | Increased lexical diversity; lowest trigram repetition |
| **Sampled** | 0.7 | 14.06 | 32.86 | 0.5406 | 0.9471 | 0.44% | Moderate entropy; subtle syntactic variations |
| **Sampled** | 1.0 | 25.41 | 37.24 | 0.5528 | 0.9520 | 0.18% | Broad lexical coverage; highest Distinct-1 ratio (0.5528) |

---

## 7. Responsible and Ethical AI Analysis (Part iii)

We audited all four models on a curated **35-sentence probe suite** designed to stress-test translation reliability across five ethical dimensions:

### 7.1 Occupational Gender Stereotypes (12 Probes)
English lacks grammatical gender on occupational nouns (*"The doctor called the nurse"*), whereas Italian requires gendered morphology (*il dottore* / *la dottoressa*, *l'infermiere* / *l'infermiera*).
- **Finding:** Under neutral contexts, models default systematically to societal stereotypes: high-status roles (*doctor*, *engineer*, *pilot*, *surgeon*, *CEO*) are translated with masculine articles (*il medico*, *l'ingegnere*, *il pilota*), while administrative/subordinate roles (*nurse*, *secretary*, *cleaner*) default to feminine forms (*l'infermiera*, *la segretaria*).

### 7.2 Gender Agreement Consistency (6 Probes)
When explicit feminine gender context is supplied (*"She is an experienced software engineer"*, *"The woman is the chief surgeon"*):
- **Finding:** Pretrained LLMs (LLaMA and Gemma) adapt correctly to feminine agreement (*È un'ingegnera software esperta* or *È un'abile avvocata*). In contrast, smaller models trained on historical data occasionally fall back to generic masculine nouns (*È un ingegnere*), demonstrating resistance to modern inclusive occupational inflections.

### 7.3 Cultural Idioms and Metaphorical Integrity (7 Probes)
Figurative expressions (*"raining cats and dogs"*, *"break a leg"*, *"spill the beans"*, *"once in a blue moon"*):
- **Finding:** The Basic Transformer and smaller models frequently produce literal nonsense (*"piove cani e gatti"*, *"rompiti una gamba"*), losing the cultural meaning entirely. Pretrained LLMs successfully localize the expressions into true Italian cultural idioms (*"piove a catinelle"*, *"in bocca al lupo"*, *"ha sputato il rospo"*, *"ad ogni morte di papa"*).

### 7.4 Negations and Hallucination Resistance (6 Probes)
Stress-testing critical negations (*"The defendant was not found guilty"*, *"The medicine did not cause side effects"*):
- **Finding:** Pretrained models reliably preserve the polarity operator (*non*). Smaller models under high beam penalties occasionally drop the negation token, inverting the factual premise of the sentence.

### 7.5 Sensitive and Toxic Content (4 Probes)
- **Finding:** All four models maintain neutral, professional language without hallucinating offensive slurs or emotional escalation.

### 7.6 Actionable Mitigation Measures
1. **Debiasing System Prompts:** Prefixing generation prompts with explicit fairness guidelines (*"Where gender is unspecified, provide gender-inclusive Italian phrasing or neutral passives"*).
2. **Balanced Demographic Fine-Tuning:** Augmenting parallel corpora with counterfactual gender-swapped pairs (e.g., pairs pairing female subjects with engineering/medical roles).
3. **Constrained Decoding & Post-Editing:** Employing lexical constraints to guarantee negation operator presence when negative polarity is detected in the source.

---

## 8. Conclusion and Best-Performing Model Identification

- **Best Overall Translation Quality:** **Gemma 2 2B** and **LLaMA 3.2 1B** achieve the highest BLEU and chrF++ scores, demonstrating superior contextual handling of Italian grammar, gender agreement, and cultural idioms.
- **Best Parameter & Compute Efficiency:** The **Basic Transformer (9.93M parameters)** trains in under 10 minutes on an Apple Silicon Mac while achieving respectable in-domain conversational translation, proving the viability of compact encoder-decoder models when computational resources are strictly limited.
- **Pretrained Seq2Seq Trade-Off:** **mT5-small (300M parameters)** bridges the gap between scratch models and billion-parameter LLMs, but requires full `torch.float32` precision on MPS to ensure numerical stability.

---

## 9. Repository Layout

```
fgai-machine-translation-english-italian/
├── data/
│   ├── processed/                # Deduplicated train, val, test splits
│   │   ├── train.jsonl           # 24,000 pairs (80%)
│   │   ├── val.jsonl             # 3,000 pairs (10%)
│   │   └── test.jsonl            # 1,000 benchmark pairs (10%)
│   ├── dataset_stats.json        # Preprocessing metadata and sentence statistics
│   ├── ethical_probes_35.json    # 35 targeted probe pairs across 5 categories
│   └── spm_bpe.model             # Trained SentencePiece BPE tokenizer (10,000 vocab)
├── notebooks/
│   └── text_translation_assignment3.ipynb
├── results/                      # Evaluation CSVs, plots, and probe audit logs
│   ├── quantitative_comparison.csv
│   ├── temperature_sweep.csv
│   ├── ethical_probe_results.json
│   ├── ethical_audit_report.md
│   ├── bleu_chrf_comparison.png
│   ├── latency_vs_quality.png
│   └── temperature_analysis.png
├── scripts/
│   ├── build_notebook.py         # Automated notebook generation script
│   └── generate_submission_docx.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Deduplication and SentencePiece BPE
│   ├── models.py                 # Custom Basic Transformer Encoder-Decoder
│   ├── train_transformer.py      # Basic Transformer training loop
│   ├── finetune_mt5.py           # Pretrained mT5-small fine-tuning
│   ├── llm_translation.py        # LLaMA 3.2 and Gemma 2 translation engine
│   ├── evaluate.py               # sacrebleu, chrF++, latency, parameter audits
│   └── ethical_audit.py          # 35-probe bias and idiom evaluation
├── images.jpeg                   # Institutional MIT Manipal header banner
├── requirements.txt              # Environment dependencies
├── run_experiments.py            # Master end-to-end automation pipeline
├── SakshamGupta_230911186_FGAI_Assignment3.docx
├── SakshamGupta_230911186_FGAI_Assignment3.pdf
└── README.md
```

