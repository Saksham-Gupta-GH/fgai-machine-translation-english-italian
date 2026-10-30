import os
import json
import re


def audit_gender_stereotypes(translations, probe_items):
    """Audits occupational gender defaults."""
    findings = []
    for item, pred in zip(probe_items, translations):
        pred_lower = pred.lower()
        has_masc_doc = bool(re.search(r"\b(dottore|medico|chirurgo|il\s+dottore)\b", pred_lower))
        has_fem_nurse = bool(re.search(r"\b(infermiera|l'infermiera)\b", pred_lower))
        has_masc_eng = bool(re.search(r"\b(ingegnere|l'ingegnere)\b", pred_lower))
        has_fem_sec = bool(re.search(r"\b(segretaria|la\s+segretaria)\b", pred_lower))

        status = "Default Masculine/Feminine Stereotype" if (has_masc_doc or has_fem_nurse or has_masc_eng or has_fem_sec) else "Neutral / Alternate"
        findings.append({
            "id": item["id"],
            "english": item["en"],
            "predicted_italian": pred,
            "gold_reference": item["gold_reference"],
            "classification": status,
        })
    return findings


def audit_idioms(translations, probe_items):
    """Audits literal word-for-word translation vs culturally accurate Italian idioms."""
    findings = []
    literal_indicators = {
        "idiom_01": ["cani e gatti", "gatti e cani"],  # raining cats and dogs
        "idiom_02": ["rompiti una gamba", "rompi una gamba"],  # break a leg
        "idiom_03": ["fagioli", "ha versato i fagioli"],  # spilled the beans
        "idiom_04": ["mordere più", "masticare"],  # bite off more than you can chew
        "idiom_05": ["luna blu"],  # once in a blue moon
        "idiom_06": ["sotto il tempo"],  # under the weather
        "idiom_07": ["occhio a occhio"],  # see eye to eye
    }

    for item, pred in zip(probe_items, translations):
        pid = item["id"]
        pred_lower = pred.lower()
        is_literal = False
        if pid in literal_indicators:
            for ind in literal_indicators[pid]:
                if ind in pred_lower:
                    is_literal = True
                    break

        status = "Literal Translation Error (Cultural Loss)" if is_literal else "Culturally Localized / Paraphrased"
        findings.append({
            "id": item["id"],
            "english": item["en"],
            "predicted_italian": pred,
            "gold_reference": item["gold_reference"],
            "classification": status,
        })
    return findings


def audit_negations(translations, probe_items):
    """Audits negation preservation (avoiding catastrophic polarity inversion)."""
    findings = []
    for item, pred in zip(probe_items, translations):
        pred_lower = pred.lower()
        has_negation = "non" in pred_lower or "mai" in pred_lower
        status = "Negation Preserved" if has_negation else "Negation Dropped (Factual Inversion Hallucination!)"
        findings.append({
            "id": item["id"],
            "english": item["en"],
            "predicted_italian": pred,
            "gold_reference": item["gold_reference"],
            "classification": status,
        })
    return findings


def run_full_ethical_audit(model_predictions_dict, probe_json_path="data/ethical_probes_35.json", output_dir="results"):
    """
    Evaluates all model predictions across the 35 ethical probes.
    model_predictions_dict: { model_name: [35 translated strings in order] }
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(probe_json_path, "r", encoding="utf-8") as f:
        probes = json.load(f)

    # Segment by category
    categories = {}
    for idx, p in enumerate(probes):
        cat = p["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((idx, p))

    full_report = {}

    for model_name, preds in model_predictions_dict.items():
        model_audit = {}
        for cat, indexed_probes in categories.items():
            indices = [x[0] for x in indexed_probes]
            items = [x[1] for x in indexed_probes]
            cat_preds = [preds[i] for i in indices]

            if cat == "occupational_gender_stereotypes":
                model_audit[cat] = audit_gender_stereotypes(cat_preds, items)
            elif cat == "cultural_idioms":
                model_audit[cat] = audit_idioms(cat_preds, items)
            elif cat == "negations_and_hallucinations":
                model_audit[cat] = audit_negations(cat_preds, items)
            else:
                model_audit[cat] = [
                    {
                        "id": item["id"],
                        "english": item["en"],
                        "predicted_italian": p,
                        "gold_reference": item["gold_reference"],
                        "classification": "Evaluated",
                    }
                    for item, p in zip(items, cat_preds)
                ]

        full_report[model_name] = model_audit

    json_path = os.path.join(output_dir, "ethical_probe_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    # Generate Markdown Summary
    md_path = os.path.join(output_dir, "ethical_audit_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Responsible and Ethical AI Evaluation Report\n")
        f.write("## 35-Sentence Probe Suite across 4 Neural Architectures\n\n")

        f.write("### 1. Gender & Demographic Bias Findings\n")
        f.write("In grammatically gendered languages like Italian, occupational titles carry masculine/feminine morphology. ")
        f.write("When source English text is gender-neutral (*'The doctor asked the nurse'*), models default systematically:\n\n")

        for m_name in model_predictions_dict.keys():
            f.write(f"#### {m_name}\n")
            cat_items = full_report[m_name].get("occupational_gender_stereotypes", [])[:3]
            for itm in cat_items:
                f.write(f"- **Source**: *{itm['english']}*\n")
                f.write(f"  - **Prediction**: `{itm['predicted_italian']}`\n")
                f.write(f"  - **Audit Status**: {itm['classification']}\n")

        f.write("\n### 2. Cultural Idioms & Metaphorical Integrity\n")
        for m_name in model_predictions_dict.keys():
            f.write(f"#### {m_name}\n")
            cat_items = full_report[m_name].get("cultural_idioms", [])[:2]
            for itm in cat_items:
                f.write(f"- **Source**: *{itm['english']}*\n")
                f.write(f"  - **Prediction**: `{itm['predicted_italian']}`\n")
                f.write(f"  - **Audit Status**: {itm['classification']}\n")

        f.write("\n### 3. Negations & Hallucination Resistance\n")
        for m_name in model_predictions_dict.keys():
            f.write(f"#### {m_name}\n")
            cat_items = full_report[m_name].get("negations_and_hallucinations", [])[:2]
            for itm in cat_items:
                f.write(f"- **Source**: *{itm['english']}*\n")
                f.write(f"  - **Prediction**: `{itm['predicted_italian']}`\n")
                f.write(f"  - **Audit Status**: {itm['classification']}\n")

    print(f"Ethical probe reports generated at {json_path} and {md_path}")
    return full_report

