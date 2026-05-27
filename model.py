import math
import torch
import torch.nn as nn
from codebook import INPUT_VOCAB_SIZE, NUM_MOVES, MAX_SEQ_LEN


class TicTacToeTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_ff=128):
        super().__init__()
        self.embedding = nn.Embedding(INPUT_VOCAB_SIZE, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, NUM_MOVES)

    def forward(self, x, legal_mask=None):
        pad_mask = x == 0
        h = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        # pool over non-padding positions
        mask = (~pad_mask).unsqueeze(-1).float()
        h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.head(h)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))
        return logits
