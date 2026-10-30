import os
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.models import BasicTransformer
from src.preprocessing import load_sentencepiece

SEED = 42
torch.manual_seed(SEED)


class ParallelTextDataset(Dataset):
    """PyTorch Dataset for parallel sentence pairs pre-tokenized via SentencePiece for maximum GPU throughput."""

    def __init__(self, jsonl_path, sp_model, max_len=64):
        self.sp = sp_model
        self.max_len = max_len
        self.data = []

        bos = sp_model.bos_id()
        eos = sp_model.eos_id()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                en_ids = [bos] + sp_model.encode(item["en"], out_type=int)[: max_len - 2] + [eos]
                it_ids = [bos] + sp_model.encode(item["it"], out_type=int)[: max_len - 2] + [eos]
                self.data.append((
                    torch.tensor(en_ids, dtype=torch.long),
                    torch.tensor(it_ids, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_idx=0):
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, clip_grad=1.0):
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing inputs and targets
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_input)

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_basic_transformer(
    data_dir="data",
    results_dir="results",
    epochs=10,
    batch_size=64,
    d_model=256,
    nhead=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    lr=5e-4,
    warmup_steps=1000,
    device=None,
):
    os.makedirs(results_dir, exist_ok=True)
    proc_dir = os.path.join(data_dir, "processed")
    train_path = os.path.join(proc_dir, "train.jsonl")
    val_path = os.path.join(proc_dir, "val.jsonl")

    sp = load_sentencepiece(os.path.join(data_dir, "spm_bpe"))
    vocab_size = sp.get_piece_size()

    train_ds = ParallelTextDataset(train_path, sp)
    val_ds = ParallelTextDataset(val_path, sp)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx=sp.pad_id())
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx=sp.pad_id())
    )

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"\n=======================================================")
    print(f"Training Basic Transformer (From Scratch)")
    print(f"  Device: {device}")
    print(f"  Vocabulary Size: {vocab_size:,}")
    print(f"  Train Pairs: {len(train_ds):,} | Val Pairs: {len(val_ds):,}")
    print(f"  Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"=======================================================\n")

    model = BasicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pad_idx=sp.pad_id(),
        bos_idx=sp.bos_id(),
        eos_idx=sp.eos_id(),
    ).to(device)

    param_stats = model.count_parameters()
    print(f"Model Parameters: {param_stats['total_parameters']:,} (100% trainable)")

    criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id(), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_model_path = os.path.join(results_dir, "basic_transformer_best.pt")
    history = []

    start_wall_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        val_ppl = math.exp(min(val_loss, 20.0))
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "config": {
                        "vocab_size": vocab_size,
                        "d_model": d_model,
                        "nhead": nhead,
                        "num_encoder_layers": num_encoder_layers,
                        "num_decoder_layers": num_decoder_layers,
                        "dim_feedforward": dim_feedforward,
                    },
                },
                best_model_path,
            )

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
    print(f"\nTraining Complete in {total_training_time/60:.2f} minutes.", flush=True)
    print(f"Best Validation Loss: {best_val_loss:.4f} saved to {best_model_path}", flush=True)

    # Save training history
    history_path = os.path.join(results_dir, "basic_transformer_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "Basic Transformer (Encoder-Decoder)",
                "total_parameters": param_stats["total_parameters"],
                "total_training_time_sec": round(total_training_time, 2),
                "best_val_loss": round(best_val_loss, 4),
                "history": history,
            },
            f,
            indent=2,
        )

    return model, best_model_path, history


if __name__ == "__main__":
    train_basic_transformer(epochs=8, batch_size=64)
