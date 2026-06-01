import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from checkers8free.codebook import encode_fen, pad_sequence, NUM_MOVES
from checkers8free.model import CheckersTransformer


class CheckersDataset(Dataset):
    def __init__(self, data):
        # Precompute tokens, dense value targets and legal masks once. The value
        # list enumerates exactly the legal moves, so the mask comes straight
        # from it — no need to regenerate legal moves from the board.
        self.tokens = []
        self.targets = []
        self.legal = []
        for i, item in enumerate(data):
            fen = item['fen']
            self.tokens.append(torch.tensor(pad_sequence(encode_fen(fen)), dtype=torch.long))

            target = torch.zeros(NUM_MOVES, dtype=torch.float)
            legal = torch.zeros(NUM_MOVES, dtype=torch.bool)
            for move_idx, val in item['values']:
                target[move_idx] = val
                legal[move_idx] = True
            self.targets.append(target)
            self.legal.append(legal)

            if (i + 1) % 2000 == 0:
                print(f'  precomputed {i + 1}/{len(data)} positions', flush=True)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.targets[idx], self.legal[idx]


def masked_mse(pred, target, legal):
    m = legal.float()
    sq = (pred - target) ** 2 * m
    return sq.sum() / m.sum().clamp(min=1)


def policy_agreement(pred, target, legal):
    """Fraction of positions where the highest-value predicted legal move is an
    optimal move (ties on the true value count as agreement)."""
    neg = torch.finfo(pred.dtype).min
    masked_pred = torch.where(legal, pred, torch.full_like(pred, neg))
    chosen = masked_pred.argmax(dim=1)
    masked_target = torch.where(legal, target, torch.full_like(target, neg))
    best_target = masked_target.max(dim=1).values
    chosen_target = target.gather(1, chosen.unsqueeze(1)).squeeze(1)
    return (torch.isclose(chosen_target, best_target, atol=1e-3)).float().mean().item()


def save_cpu_state(model, path):
    # Always persist CPU tensors so the Flask app loads cleanly on a CPU-only
    # machine even when the weights were trained on a GPU (e.g. Colab / MPS).
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def train(epochs=300, lr=1e-3, batch_size=512, data_path='checkers8free/data.json',
          out_path='checkers8free/model.pt', device=None, resume=False):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f'Device: {device}')

    with open(data_path) as f:
        data = json.load(f)

    print(f'Training on {len(data)} positions')
    dataset = CheckersDataset(data)
    pin = device == 'cuda'
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=pin)

    model = CheckersTransformer().to(device)
    if resume and os.path.exists(out_path):
        model.load_state_dict(torch.load(out_path, map_location=device))
        print(f'Warm-started from {out_path}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_agree = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0
        agree_sum = 0
        total = 0
        model.train()
        for x, y, legal in loader:
            x, y, legal = x.to(device), y.to(device), legal.to(device)
            pred = model(x)  # raw values, no masking — loss is masked instead
            loss = masked_mse(pred, y, legal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            agree_sum += policy_agreement(pred.detach(), y, legal) * bs
            total += bs

        agree = agree_sum / total
        avg_loss = total_loss / total
        improved = agree > best_agree
        if improved:
            best_agree = agree
            save_cpu_state(model, out_path)

        print(f'Epoch {epoch:3d}/{epochs} | MSE: {avg_loss:.5f} | '
              f'Policy agreement: {agree:.4f} | Best: {best_agree:.4f}'
              f"{'  *saved' if improved else ''}", flush=True)

    save_cpu_state(model, out_path)
    print(f'Model saved to {out_path}. Best policy agreement: {best_agree:.4f}')
    return model


if __name__ == '__main__':
    train()
