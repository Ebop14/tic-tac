import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from checkers8free.codebook import encode_fen, pad_sequence, encode_move, NUM_MOVES
from checkers8free.notation import fen_to_board
from checkers8free.model import CheckersTransformer


class CheckersDataset(Dataset):
    def __init__(self, data):
        # The token ids and legal-move masks are deterministic per position, so
        # precompute them once instead of regenerating legal moves for every
        # item on every epoch (that Python work dominated each epoch's runtime).
        self.tokens = []
        self.targets = []
        self.legal = []
        for i, item in enumerate(data):
            fen = item['fen']
            self.tokens.append(torch.tensor(pad_sequence(encode_fen(fen)), dtype=torch.long))
            self.targets.append(torch.tensor(item['move_idx'], dtype=torch.long))

            board = fen_to_board(fen)
            legal = torch.zeros(NUM_MOVES, dtype=torch.bool)
            for move in board.legal_moves():
                origin, steps = move
                legal[encode_move(origin, steps[-1])] = True
            self.legal.append(legal)

            if (i + 1) % 2000 == 0:
                print(f'  precomputed {i + 1}/{len(data)} positions', flush=True)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.targets[idx], self.legal[idx]


def save_cpu_state(model, path):
    # Always persist CPU tensors so the Flask app loads cleanly on a CPU-only
    # machine even when the weights were trained on a GPU (e.g. Colab).
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def train(epochs=200, lr=1e-3, batch_size=512, data_path='checkers8free/data.json',
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
    # __getitem__ is now a trivial tensor lookup, so worker processes add only
    # fork overhead; keep it single-process.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=pin)

    model = CheckersTransformer().to(device)
    if resume and os.path.exists(out_path):
        model.load_state_dict(torch.load(out_path, map_location=device))
        print(f'Warm-started from {out_path}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, legal_mask=mask)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        improved = acc > best_acc
        if improved:
            best_acc = acc
            save_cpu_state(model, out_path)

        print(f'Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | '
              f'Accuracy: {acc:.4f} | Best: {best_acc:.4f}'
              f"{'  *saved' if improved else ''}", flush=True)

    save_cpu_state(model, out_path)
    print(f'Model saved to {out_path}. Best accuracy: {best_acc:.4f}')
    return model


if __name__ == '__main__':
    train()
