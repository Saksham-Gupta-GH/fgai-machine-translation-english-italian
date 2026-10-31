import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding as proposed in Vaswani et al. (2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class BasicTransformer(nn.Module):
    """
    Sequence-to-Sequence Transformer with classic Encoder-Decoder architecture.
    Multi-head self-attention in encoder, multi-head self-attention and cross-attention in decoder.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 2,
        eos_idx: int = 3,
        max_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Shared embedding for input/output subword tokens
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying between embedding and final linear projection
        self.generator.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Causal triangular mask ensuring position i only attends to positions <= i."""
        return nn.Transformer.generate_square_subsequent_mask(sz, device=device).bool()

    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor, device: torch.device):
        tgt_seq_len = tgt.size(1)

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        src_padding_mask = (src == self.pad_idx).to(device)
        tgt_padding_mask = (tgt == self.pad_idx).to(device)

        return None, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        device = src.device
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_masks(src, tgt, device)

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(out)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_padding_mask: torch.Tensor = None):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ):
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    @torch.no_grad()
    def translate_greedy(self, src_ids: torch.Tensor, max_len: int = 64) -> torch.Tensor:
        """
        Translates a single source sequence (shape: [1, src_len]) or batch via greedy search.
        Encodes once, then autoregressively decodes token-by-token.
        """
        self.eval()
        device = src_ids.device
        batch_size = src_ids.size(0)

        src_padding_mask = (src_ids == self.pad_idx).to(device)
        memory = self.encode(src_ids, src_padding_mask=src_padding_mask)

        # Initialize target tensor with <bos>
        ys = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1), device)
            out = self.decode(
                ys,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            prob = self.generator(out[:, -1])  # Predict next token from the last position
            next_word = torch.argmax(prob, dim=1)
            next_word = torch.where(finished, self.eos_idx, next_word)

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Check if all sequences in batch hit <eos>
            finished |= (next_word == self.eos_idx)
            if finished.all():
                break

        return ys

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


if __name__ == "__main__":
    model = BasicTransformer(vocab_size=10000, d_model=256, nhead=4, num_encoder_layers=4, num_decoder_layers=4)
    params = model.count_parameters()
    print(f"Basic Transformer initialized successfully:")
    print(f"  Total Parameters: {params['total_parameters']:,}")
    print(f"  Trainable Parameters: {params['trainable_parameters']:,}")
    
    # Test forward pass with dummy inputs
    src = torch.randint(4, 1000, (2, 10))
    tgt = torch.randint(4, 1000, (2, 8))
    logits = model(src, tgt)
    print(f"Logits shape: {logits.shape}")  # (2, 8, 10000)
    
    # Test greedy translation
    out_ids = model.translate_greedy(src[:1])
    print(f"Greedy translation output shape: {out_ids.shape}")
