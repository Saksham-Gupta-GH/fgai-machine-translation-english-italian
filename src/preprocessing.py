import os
import json
import re
import random
import unicodedata
from collections import OrderedDict
from datasets import load_dataset
import sentencepiece as spm

SEED = 42
random.seed(SEED)


def clean_text(text: str) -> str:
    """Normalizes whitespace, removes leading dashes/bullets, and cleans punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Strip leading dialogue dashes or bullet symbols
    text = re.sub(r"^[-–—•*\s]+", "", text)
    # Remove HTML/XML markup tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_pair(en: str, it: str, min_words=4, max_words=35, max_ratio=2.2) -> bool:
    """Filters noisy, too-short, too-long, or heavily misaligned pairs."""
    if not en or not it:
        return False
    # Reject uppercase headers or table markers (e.g. 'ANNEX I', 'TABLE 4')
    if en.isupper() and len(en.split()) <= 3:
        return False
    words_en = en.split()
    words_it = it.split()
    len_en = len(words_en)
    len_it = len(words_it)
    
    if len_en < min_words or len_en > max_words:
        return False
    if len_it < min_words or len_it > max_words:
        return False
        
    ratio = max(len_en, len_it) / max(min(len_en, len_it), 1)
    if ratio > max_ratio:
        return False
        
    return True


def prepare_parallel_dataset(
    dataset_name="Helsinki-NLP/opus-100",
    output_dir="data",
    train_size=24000,
    val_size=3000,
    test_size=1000,
    seed=SEED
):
    """
    Downloads, deduplicates by English source, length-filters, and splits
    the parallel corpus into Train, Validation, and Test sets with zero leakage.
    """
    raw_dir = os.path.join(output_dir, "raw")
    proc_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    train_path = os.path.join(proc_dir, "train.jsonl")
    val_path = os.path.join(proc_dir, "val.jsonl")
    test_path = os.path.join(proc_dir, "test.jsonl")
    stats_path = os.path.join(output_dir, "dataset_stats.json")

    # If already prepared, return file paths
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"Parallel dataset already prepared at: {proc_dir}")
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return train_path, val_path, test_path, stats

    print(f"Loading '{dataset_name}' (en-it) from Hugging Face...")
    ds = load_dataset(dataset_name, "en-it")

    # Combine pool to apply rigorous global source-level deduplication
    candidates = []
    print("Collecting and normalizing parallel sentence candidates...")
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            for item in ds[split_name]:
                trans = item.get("translation", {})
                en_raw = trans.get("en", "")
                it_raw = trans.get("it", "")
                en_clean = clean_text(en_raw)
                it_clean = clean_text(it_raw)
                if is_valid_pair(en_clean, it_clean):
                    candidates.append((en_clean, it_clean))

    print(f"Total raw candidates meeting length criteria: {len(candidates)}")

    # Source-level deduplication: Group by lowercased English source sentence
    # Retaining one clean canonical Italian target per unique English source sentence
    print("Applying strict source-level deduplication (zero train-test leakage guarantee)...")
    unique_pairs = OrderedDict()
    for en, it in candidates:
        en_norm = en.lower()
        if en_norm not in unique_pairs:
            unique_pairs[en_norm] = (en, it)

    deduped_list = list(unique_pairs.values())
    print(f"Unique source-sentence parallel pairs available: {len(deduped_list)}")

    # Unbiased random shuffle with fixed seed
    random.seed(seed)
    random.shuffle(deduped_list)

    total_needed = train_size + val_size + test_size
    if len(deduped_list) < total_needed:
        raise ValueError(f"Available unique pairs ({len(deduped_list)}) < requested total ({total_needed})")

    selected_pool = deduped_list[:total_needed]
    train_pairs = selected_pool[:train_size]
    val_pairs = selected_pool[train_size : train_size + val_size]
    test_pairs = selected_pool[train_size + val_size : total_needed]

    def save_jsonl(pairs, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for idx, (en, it) in enumerate(pairs):
                line = json.dumps({"id": idx, "en": en, "it": it}, ensure_ascii=False)
                f.write(line + "\n")

    save_jsonl(train_pairs, train_path)
    save_jsonl(val_pairs, val_path)
    save_jsonl(test_pairs, test_path)

    # Compute length and lexical statistics
    all_en_words = [len(p[0].split()) for p in selected_pool]
    all_it_words = [len(p[1].split()) for p in selected_pool]

    stats = {
        "dataset_name": dataset_name,
        "dataset_url": f"https://huggingface.co/datasets/{dataset_name}",
        "language_pair": "en-it",
        "random_seed": seed,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "total_selected_pairs": total_needed,
        "mean_en_words": round(sum(all_en_words) / len(all_en_words), 2),
        "mean_it_words": round(sum(all_it_words) / len(all_it_words), 2),
        "max_en_words": max(all_en_words),
        "max_it_words": max(all_it_words),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Dataset successfully prepared and partitioned into {proc_dir}:")
    print(f"  - Train: {len(train_pairs)} pairs")
    print(f"  - Validation: {len(val_pairs)} pairs")
    print(f"  - Test: {len(test_pairs)} pairs")
    print(f"  - Statistics saved to {stats_path}")

    return train_path, val_path, test_path, stats


def train_sentencepiece_bpe(
    train_jsonl_path,
    model_prefix="data/spm_bpe",
    vocab_size=10000,
    character_coverage=1.0
):
    """
    Trains a shared subword BPE tokenizer using SentencePiece strictly on the training partition.
    Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3.
    """
    model_file = f"{model_prefix}.model"
    if os.path.exists(model_file):
        print(f"SentencePiece model already exists at: {model_file}")
        sp = spm.SentencePieceProcessor()
        sp.load(model_file)
        return sp

    # Create temporary text file containing training sentences (both English and Italian)
    temp_corpus = f"{model_prefix}_train_corpus.txt"
    with open(train_jsonl_path, "r", encoding="utf-8") as fin, open(temp_corpus, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            fout.write(item["en"] + "\n")
            fout.write(item["it"] + "\n")

    print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size}) strictly on train set...")
    spm.SentencePieceTrainer.train(
        input=temp_corpus,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
    )

    if os.path.exists(temp_corpus):
        os.remove(temp_corpus)

    print(f"SentencePiece model saved to: {model_file}")
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp


def load_sentencepiece(model_prefix="data/spm_bpe"):
    """Loads existing SentencePiece tokenizer."""
    model_file = f"{model_prefix}.model"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"SentencePiece model not found at {model_file}. Run train_sentencepiece_bpe first.")
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp


if __name__ == "__main__":
    # Test execution
    train_f, val_f, test_f, stats = prepare_parallel_dataset()
    sp = train_sentencepiece_bpe(train_f, vocab_size=10000)
    sample_text = "The quick brown fox jumps over the lazy dog."
    encoded = sp.encode(sample_text, out_type=str)
    print(f"Sample encode: {encoded}")
    print(f"Token IDs: {sp.encode(sample_text, out_type=int)}")
    print(f"Decoded: {sp.decode(encoded)}")

