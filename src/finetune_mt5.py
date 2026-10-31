import os
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, MT5ForConditionalGeneration

SEED = 42
torch.manual_seed(SEED)
TASK_PREFIX = "translate English to Italian: "


class MT5TranslationDataset(Dataset):
    """Dataset for MT5 Seq2Seq fine-tuning with explicit task prefix."""

    def __init__(self, jsonl_path, tokenizer, max_src_len=64, max_tgt_len=64):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append((TASK_PREFIX + item["en"], item["it"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        return src_text, tgt_text


def collate_fn(batch, tokenizer, max_src_len=64, max_tgt_len=64):
    src_texts, tgt_texts = zip(*batch)

    model_inputs = tokenizer(
        list(src_texts),
        max_length=max_src_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    labels = tokenizer(
        list(tgt_texts),
        max_length=max_tgt_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    # Replace padding token id with -100 so it is ignored by CrossEntropyLoss
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def train_epoch(model, dataloader, optimizer, scheduler, device, clip_grad=1.0):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def finetune_mt5_small(
    model_name="google/mt5-small",
    data_dir="data",
    results_dir="results",
    epochs=3,
    batch_size=32,
    lr=3e-4,
    warmup_steps=500,
    device=None,
):
    os.makedirs(results_dir, exist_ok=True)
    proc_dir = os.path.join(data_dir, "processed")
    train_path = os.path.join(proc_dir, "train.jsonl")
    val_path = os.path.join(proc_dir, "val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = MT5TranslationDataset(train_path, tokenizer)
    val_ds = MT5TranslationDataset(val_path, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"\n=======================================================")
    print(f"Fine-Tuning mT5-small (Pretrained Encoder-Decoder)")
    print(f"  Device: {device}")
    print(f"  Precision: Strictly float32 (eliminates MPS NaN overflow)")
    print(f"  Train Pairs: {len(train_ds):,} | Val Pairs: {len(val_ds):,}")
    print(f"  Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"=======================================================\n")

    # Strictly torch.float32 to avoid NaN overflow in mT5 LayerNorm
    model = MT5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32, tie_word_embeddings=False).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    save_dir = os.path.join(results_dir, "mt5_small_best")
    os.makedirs(save_dir, exist_ok=True)
    history = []

    start_wall_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate_epoch(model, val_loader, device)
        epoch_time = time.time() - t0

        val_ppl = math.exp(min(val_loss, 20.0))
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 2),
                "epoch_time_sec": round(epoch_time, 2),
                "is_best": is_best,
            }
        )

        marker = "🏆 (Best)" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Time: {epoch_time:.1f}s {marker}",
            flush=True,
        )

    total_training_time = time.time() - start_wall_time
    print(f"\nmT5 Fine-Tuning Complete in {total_training_time/60:.2f} minutes.", flush=True)
    print(f"Best Validation Loss: {best_val_loss:.4f} saved to {save_dir}", flush=True)

    history_path = os.path.join(results_dir, "mt5_small_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "mT5-small (Pretrained Encoder-Decoder)",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "total_training_time_sec": round(total_training_time, 2),
                "best_val_loss": round(best_val_loss, 4),
                "history": history,
            },
            f,
            indent=2,
        )

    return model, tokenizer, save_dir, history


@torch.no_grad()
def translate_mt5(model, tokenizer, texts, device, max_length=64, do_sample=False, num_beams=1, temperature=1.0):
    """Generates translations using mT5 under comparable decoding settings."""
    model.eval()
    prefixed_texts = [TASK_PREFIX + t for t in texts]
    inputs = tokenizer(prefixed_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(
        device
    )

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_length,
        **({"temperature": temperature} if do_sample else {}),
        do_sample=do_sample,
        num_beams=num_beams,
    )
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


if __name__ == "__main__":
    finetune_mt5_small(epochs=2, batch_size=32)
