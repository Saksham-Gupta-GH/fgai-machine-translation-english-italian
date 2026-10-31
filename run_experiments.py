import os
import sys
import json
import time
import torch
import pandas as pd

from src.preprocessing import prepare_parallel_dataset, train_sentencepiece_bpe, load_sentencepiece
from src.models import BasicTransformer
from src.train_transformer import train_basic_transformer
from src.finetune_mt5 import finetune_mt5_small, translate_mt5
from src.llm_translation import load_llama_model, load_gemma_model, batch_translate_llm
from src.evaluate import evaluate_translations, plot_metric_comparisons, plot_temperature_sweep
from src.ethical_audit import run_full_ethical_audit
from scripts.generate_submission_docx import build_submission_docx

SEED = 42
torch.manual_seed(SEED)


def run_full_pipeline(
    data_dir="data",
    results_dir="results",
    epochs_transformer=8,
    epochs_mt5=2,
    batch_size_transformer=64,
    batch_size_mt5=32,
    test_eval_limit=1000,
    temp_sweep_limit=300,
):
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n==================================================================")
    print(f"FGAI Assignment 3: Machine Translation Master Pipeline (en -> it)")
    print(f"  Device: {device}")
    print(f"  Results Directory: {results_dir}")
    print(f"==================================================================\n")

    # Step 1: Preprocessing & Tokenization
    print("\n--- [Step 1/6] Dataset Preparation & Deduplication ---")
    train_path, val_path, test_path, stats = prepare_parallel_dataset(output_dir=data_dir, seed=SEED)
    sp = load_sentencepiece(os.path.join(data_dir, "spm_bpe"))
    print(f"Loaded SentencePiece vocabulary ({sp.get_piece_size():,} subwords).")

    # Load test sentences
    test_en = []
    test_it_gold = []
    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= test_eval_limit:
                break
            item = json.loads(line)
            test_en.append(item["en"])
            test_it_gold.append(item["it"])

    print(f"Loaded {len(test_en)} benchmark test sentence pairs for evaluation.")

    # Load 35 ethical probes
    probes_path = os.path.join(data_dir, "ethical_probes_35.json")
    with open(probes_path, "r", encoding="utf-8") as f:
        ethical_probes = json.load(f)
    probe_en = [p["en"] for p in ethical_probes]

    model_results_table = []
    model_predictions = {}
    model_probe_predictions = {}

    # Step 2: Model 1 — Basic Transformer (From Scratch)
    print("\n--- [Step 2/6] Training & Evaluating Model 1: Basic Transformer ---")
    transformer_model, trans_ckpt_path, trans_hist = train_basic_transformer(
        data_dir=data_dir,
        results_dir=results_dir,
        epochs=epochs_transformer,
        batch_size=batch_size_transformer,
        device=device,
    )

    # Translate test set with Basic Transformer (Greedy)
    print("Translating test set with Basic Transformer (Greedy)...")
    transformer_model.eval()
    trans_preds = []
    t0 = time.time()
    for en_text in test_en:
        en_ids = [sp.bos_id()] + sp.encode(en_text, out_type=int) + [sp.eos_id()]
        src_t = torch.tensor([en_ids], dtype=torch.long, device=device)
        out_ids = transformer_model.translate_greedy(src_t, max_len=64)[0].tolist()
        # Decode removing <bos> and <eos>
        pred_text = sp.decode([tid for tid in out_ids if tid not in (sp.bos_id(), sp.eos_id(), sp.pad_id())])
        trans_preds.append(pred_text)

    trans_latency_ms = ((time.time() - t0) / len(test_en)) * 1000.0
    model_predictions["Basic Transformer (Scratch)"] = trans_preds

    # Evaluate ethical probes
    trans_probe_preds = []
    for en_text in probe_en:
        en_ids = [sp.bos_id()] + sp.encode(en_text, out_type=int) + [sp.eos_id()]
        src_t = torch.tensor([en_ids], dtype=torch.long, device=device)
        out_ids = transformer_model.translate_greedy(src_t, max_len=64)[0].tolist()
        pred_text = sp.decode([tid for tid in out_ids if tid not in (sp.bos_id(), sp.eos_id(), sp.pad_id())])
        trans_probe_preds.append(pred_text)
    model_probe_predictions["Basic Transformer (Scratch)"] = trans_probe_preds

    trans_metrics = evaluate_translations(trans_preds, test_it_gold)
    trans_params = transformer_model.count_parameters()

    with open(os.path.join(results_dir, "basic_transformer_history.json"), "r") as f:
        hist_data = json.load(f)

    model_results_table.append(
        {
            "model_name": "Basic Transformer (Scratch)",
            "paradigm": "Encoder-Decoder",
            "total_params": trans_params["total_parameters"],
            "trainable_params": trans_params["trainable_parameters"],
            "training_time_min": round(hist_data["total_training_time_sec"] / 60.0, 2),
            "latency_ms": round(trans_latency_ms, 2),
            "bleu": trans_metrics["bleu"],
            "chrf": trans_metrics["chrf"],
            "chrf_pp": trans_metrics["chrf_pp"],
        }
    )

    # Step 3: Model 2 — mT5-small (Pretrained Encoder-Decoder)
    print("\n--- [Step 3/6] Fine-Tuning & Evaluating Model 2: mT5-small ---")
    mt5_model, mt5_tok, mt5_dir, mt5_hist = finetune_mt5_small(
        data_dir=data_dir,
        results_dir=results_dir,
        epochs=epochs_mt5,
        batch_size=batch_size_mt5,
        device=device,
    )

    print("Translating test set with mT5-small (Greedy, do_sample=False)...")
    t0 = time.time()
    # Batch translation for faster evaluation
    batch_size_eval = 1
    mt5_preds = []
    for i in range(0, len(test_en), batch_size_eval):
        batch_chunk = test_en[i : i + batch_size_eval]
        preds = translate_mt5(mt5_model, mt5_tok, batch_chunk, device, do_sample=False)
        mt5_preds.extend(preds)

    mt5_latency_ms = ((time.time() - t0) / len(test_en)) * 1000.0
    model_predictions["mT5-small (Finetuned)"] = mt5_preds

    # Evaluate ethical probes
    mt5_probe_preds = translate_mt5(mt5_model, mt5_tok, probe_en, device, do_sample=False)
    model_probe_predictions["mT5-small (Finetuned)"] = mt5_probe_preds

    mt5_metrics = evaluate_translations(mt5_preds, test_it_gold)
    with open(os.path.join(results_dir, "mt5_small_history.json"), "r") as f:
        mt5_hist_data = json.load(f)

    model_results_table.append(
        {
            "model_name": "mT5-small (Finetuned)",
            "paradigm": "Encoder-Decoder",
            "total_params": mt5_hist_data["total_parameters"],
            "trainable_params": mt5_hist_data["trainable_parameters"],
            "training_time_min": round(mt5_hist_data["total_training_time_sec"] / 60.0, 2),
            "latency_ms": round(mt5_latency_ms, 2),
            "bleu": mt5_metrics["bleu"],
            "chrf": mt5_metrics["chrf"],
            "chrf_pp": mt5_metrics["chrf_pp"],
        }
    )

    # Step 4: Model 3 & 4 — Decoder-Only LLMs (LLaMA 3.2 1B & Gemma 2 2B)
    print("\n--- [Step 4/6] Evaluating Models 3 & 4: LLaMA 3.2 1B & Gemma 2 2B ---")
    try:
        llama_model, llama_tok = load_llama_model(device=device)
        print("Translating benchmark test set with LLaMA 3.2 1B (Few-Shot, Greedy)...")
        llama_preds, llama_latency = batch_translate_llm(
            llama_model, llama_tok, test_en, model_type="llama", device=device, do_sample=False
        )
        model_predictions["LLaMA 3.2 1B (Few-Shot)"] = llama_preds

        llama_probe_preds, _ = batch_translate_llm(
            llama_model, llama_tok, probe_en, model_type="llama", device=device, do_sample=False
        )
        model_probe_predictions["LLaMA 3.2 1B (Few-Shot)"] = llama_probe_preds

        llama_metrics = evaluate_translations(llama_preds, test_it_gold)
        total_p = sum(p.numel() for p in llama_model.parameters())
        model_results_table.append(
            {
                "model_name": "LLaMA 3.2 1B (Few-Shot)",
                "paradigm": "Decoder-Only",
                "total_params": total_p,
                "trainable_params": 0,
                "training_time_min": "N/A (Prompted)",
                "latency_ms": round(llama_latency, 2),
                "bleu": llama_metrics["bleu"],
                "chrf": llama_metrics["chrf"],
                "chrf_pp": llama_metrics["chrf_pp"],
            }
        )
    except Exception as e:
        print(f"Warning: LLaMA loading error (check gated access or offline cache): {e}")

    try:
        gemma_model, gemma_tok = load_gemma_model(device=device)
        print("Translating benchmark test set with Gemma 2 2B (Few-Shot, Greedy)...")
        gemma_preds, gemma_latency = batch_translate_llm(
            gemma_model, gemma_tok, test_en, model_type="gemma", device=device, do_sample=False
        )
        model_predictions["Gemma 2 2B (Few-Shot)"] = gemma_preds

        gemma_probe_preds, _ = batch_translate_llm(
            gemma_model, gemma_tok, probe_en, model_type="gemma", device=device, do_sample=False
        )
        model_probe_predictions["Gemma 2 2B (Few-Shot)"] = gemma_probe_preds

        gemma_metrics = evaluate_translations(gemma_preds, test_it_gold)
        total_p = sum(p.numel() for p in gemma_model.parameters())
        model_results_table.append(
            {
                "model_name": "Gemma 2 2B (Few-Shot)",
                "paradigm": "Decoder-Only",
                "total_params": total_p,
                "trainable_params": 0,
                "training_time_min": "N/A (Prompted)",
                "latency_ms": round(gemma_latency, 2),
                "bleu": gemma_metrics["bleu"],
                "chrf": gemma_metrics["chrf"],
                "chrf_pp": gemma_metrics["chrf_pp"],
            }
        )
    except Exception as e:
        print(f"Warning: Gemma loading error (check gated access or offline cache): {e}")

    # Step 5: Sampling Temperature Sweep on 300-sentence subset
    print("\n--- [Step 5/6] Sampling Temperature Sweep (T in {0.2, 0.5, 0.7, 1.0}) ---")
    sweep_subset_en = test_en[:temp_sweep_limit]
    sweep_subset_it = test_it_gold[:temp_sweep_limit]
    sweep_results = []

    # If LLaMA is loaded, run sweep on LLaMA; else run on Basic Transformer / mT5
    sweep_model = llama_model if "llama_model" in locals() else mt5_model
    sweep_tok = llama_tok if "llama_tok" in locals() else mt5_tok
    sweep_model_type = "llama" if "llama_model" in locals() else "mt5"

    for temp in [0.2, 0.5, 0.7, 1.0]:
        print(f"Running sweep at Temperature T = {temp} (Seed = {SEED})...")
        if sweep_model_type == "llama":
            s_preds, _ = batch_translate_llm(
                sweep_model,
                sweep_tok,
                sweep_subset_en,
                model_type="llama",
                device=device,
                do_sample=True,
                temperature=temp,
            )
        else:
            s_preds = translate_mt5(sweep_model, sweep_tok, sweep_subset_en, device, do_sample=True, temperature=temp)

        ev = evaluate_translations(s_preds, sweep_subset_it)
        sweep_results.append(
            {
                "temperature": temp,
                "bleu": ev["bleu"],
                "chrf_pp": ev["chrf_pp"],
                "distinct_1": ev["distinct_1"],
                "distinct_2": ev["distinct_2"],
                "trigram_repetition_pct": ev["trigram_repetition_pct"],
            }
        )

    sweep_csv = os.path.join(results_dir, "temperature_sweep.csv")
    pd.DataFrame(sweep_results).to_csv(sweep_csv, index=False)
    plot_temperature_sweep(sweep_results, output_dir=results_dir)

    # Step 6: Ethical Audit & Final Deliverables
    print("\n--- [Step 6/6] Generating Ethical AI Audit & Submission Artifacts ---")
    run_full_ethical_audit(model_probe_predictions, probe_json_path=probes_path, output_dir=results_dir)

    # Save quantitative comparison table
    df_results = pd.DataFrame(model_results_table)
    csv_path = os.path.join(results_dir, "quantitative_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nFinal Comparative Results Table saved to {csv_path}:")
    print(df_results.to_string(index=False))

    plot_metric_comparisons(model_results_table, output_dir=results_dir)

    # Persist generated text so later metric corrections never require retraining.
    with open(os.path.join(results_dir, "test_predictions.json"), "w") as f:
        json.dump({"source": test_en, "references": test_it_gold, "predictions": model_predictions}, f, ensure_ascii=False)
    if len(model_results_table) != 4:
        raise RuntimeError("Incomplete required comparison: LLaMA or Gemma failed; partial results saved")

    # Build Word document
    # Submission documents are maintained separately.
    print("\n=======================================================")
    print("Master Pipeline Successfully Completed!")
    print("=======================================================\n")


if __name__ == "__main__":
    run_full_pipeline()

