import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from codebook import encode_fen, pad_sequence, MAX_SEQ_LEN, NUM_MOVES
from notation import fen_to_board
from model import TicTacToeTransformer


class TTTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        fen = item['fen']
        values = item['values']
        token_ids = pad_sequence(encode_fen(fen))
        board = fen_to_board(fen)
        legal = [False] * NUM_MOVES
        for m in board.legal_moves():
            legal[m] = True
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(values, dtype=torch.float),
            torch.tensor(legal, dtype=torch.bool),
        )


def masked_mse(pred, target, legal):
    """MSE over legal moves only (illegal predictions are unconstrained)."""
    m = legal.float()
    sq = (pred - target) ** 2 * m
    return sq.sum() / m.sum().clamp(min=1)


def policy_agreement(pred, target, legal):
    """Fraction of positions where the highest-value predicted legal move is
    actually an optimal move (ties on the true value count as agreement)."""
    neg = torch.finfo(pred.dtype).min
    masked_pred = torch.where(legal, pred, torch.full_like(pred, neg))
    chosen = masked_pred.argmax(dim=1)
    masked_target = torch.where(legal, target, torch.full_like(target, neg))
    best_target = masked_target.max(dim=1).values
    chosen_target = target.gather(1, chosen.unsqueeze(1)).squeeze(1)
    return (torch.isclose(chosen_target, best_target, atol=1e-4)).float().mean().item()


def train(epochs=800, lr=1e-3, batch_size=64):
    with open('data.json') as f:
        data = json.load(f)

    dataset = TTTDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TicTacToeTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_agree = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0
        agree_sum = 0
        total = 0
        model.train()
        for x, y, legal in loader:
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
        if agree > best_agree:
            best_agree = agree
            torch.save(model.state_dict(), 'model.pt')

        if epoch % 25 == 0 or epoch == 1:
            avg_loss = total_loss / total
            print(f'Epoch {epoch:3d} | MSE: {avg_loss:.5f} | Policy agreement: {agree:.4f}')

    torch.save(model.state_dict(), 'model.pt')
    print(f'Model saved to model.pt (best policy agreement: {best_agree:.4f})')
    return model


if __name__ == '__main__':
    train()
