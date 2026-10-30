import os
import json
from collections import Counter
import sacrebleu
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def compute_bleu(predictions, references):
    """Computes case-sensitive corpus BLEU score via sacrebleu."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return round(bleu.score, 2)


def compute_chrf(predictions, references):
    """Computes standard chrF (character n-gram F-score, word_order=0)."""
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=0)
    return round(chrf.score, 2)


def compute_chrf_plus_plus(predictions, references):
    """Computes chrF++ (character n-grams + word bigrams, word_order=2)."""
    chrf_pp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    return round(chrf_pp.score, 2)


def compute_diversity_metrics(texts):
    """Computes Distinct-1, Distinct-2, and Trigram Repetition Rate."""
    unigrams = []
    bigrams = []
    trigrams = []
    repeated_trigram_count = 0
    total_trigrams = 0

    for sent in texts:
        words = sent.split()
        unigrams.extend(words)

        b_grams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
        bigrams.extend(b_grams)

        t_grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        trigrams.extend(t_grams)

        counts = Counter(t_grams)
        repeated_trigram_count += sum(c - 1 for c in counts.values() if c > 1)
        total_trigrams += len(t_grams)

    distinct_1 = len(set(unigrams)) / max(len(unigrams), 1)
    distinct_2 = len(set(bigrams)) / max(len(bigrams), 1)
    trigram_rep_pct = (repeated_trigram_count / max(total_trigrams, 1)) * 100.0

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "trigram_repetition_pct": round(trigram_rep_pct, 2),
    }


def evaluate_translations(predictions, references):
    """Computes all core translation quality metrics."""
    return {
        "bleu": compute_bleu(predictions, references),
        "chrf": compute_chrf(predictions, references),
        "chrf_pp": compute_chrf_plus_plus(predictions, references),
        **compute_diversity_metrics(predictions),
    }


def plot_metric_comparisons(results_data, output_dir="results"):
    """
    Generates academic publication-ready comparison plots:
    1. BLEU and chrF comparison across architectures
    2. Latency vs. BLEU Pareto frontier
    """
    os.makedirs(output_dir, exist_ok=True)
    models = [r["model_name"] for r in results_data]
    bleu_scores = [r["bleu"] for r in results_data]
    chrf_scores = [r["chrf_pp"] for r in results_data]
    latencies = [r["latency_ms"] for r in results_data]

    # 1. Bar Chart: BLEU & chrF++ Comparison
    plt.figure(figsize=(10, 5))
    x = range(len(models))
    width = 0.35

    plt.bar([i - width / 2 for i in x], bleu_scores, width=width, label="BLEU Score", color="#2b5c8f")
    plt.bar([i + width / 2 for i in x], chrf_scores, width=width, label="chrF++ Score", color="#d95f02")

    plt.xlabel("Model Architecture & Adaptation", fontweight="bold", fontsize=11)
    plt.ylabel("Evaluation Score (0–100)", fontweight="bold", fontsize=11)
    plt.title("Comparative Translation Quality: English to Italian Benchmark", fontweight="bold", fontsize=13)
    plt.xticks(x, models, rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "bleu_chrf_comparison.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # 2. Scatter Plot: Latency vs. BLEU (Quality-Latency Trade-Off)
    plt.figure(figsize=(8, 5))
    for r in results_data:
        plt.scatter(r["latency_ms"], r["bleu"], s=180, alpha=0.85, label=r["model_name"])
        plt.text(r["latency_ms"] * 1.02, r["bleu"] + 0.3, r["model_name"], fontsize=9)

    plt.xlabel("Inference Latency per Sentence (ms, batch_size=1)", fontweight="bold", fontsize=11)
    plt.ylabel("BLEU Score", fontweight="bold", fontsize=11)
    plt.title("Inference Efficiency vs. Translation Quality (Pareto Frontier)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "latency_vs_quality.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    print(f"Generated comparison plots saved to {output_dir}:")
    print(f"  - {bar_path}")
    print(f"  - {scatter_path}")


def plot_temperature_sweep(sweep_data, output_dir="results"):
    """Plots BLEU, distinct-1, and repetition percentage across temperatures."""
    os.makedirs(output_dir, exist_ok=True)
    temps = [s["temperature"] for s in sweep_data]
    bleus = [s["bleu"] for s in sweep_data]
    d1 = [s["distinct_1"] for s in sweep_data]
    rep = [s["trigram_repetition_pct"] for s in sweep_data]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = "#1f77b4"
    ax1.set_xlabel("Sampling Temperature (T)", fontweight="bold", fontsize=11)
    ax1.set_ylabel("BLEU Score", color=color, fontweight="bold", fontsize=11)
    ax1.plot(temps, bleus, color=color, marker="o", linewidth=2.2, label="BLEU Score")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#e377c2"
    ax2.set_ylabel("Trigram Repetition (%)", color=color, fontweight="bold", fontsize=11)
    ax2.plot(temps, rep, color=color, marker="s", linestyle="--", linewidth=2.0, label="Trigram Repetition (%)")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Sampling Temperature Sweep: Translation Quality vs. Repetition", fontweight="bold", fontsize=12)
    fig.tight_layout()
    sweep_path = os.path.join(output_dir, "temperature_analysis.png")
    plt.savefig(sweep_path, dpi=300)
    plt.close()
    print(f"Temperature sweep plot saved to: {sweep_path}")

