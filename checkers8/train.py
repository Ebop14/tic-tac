import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from checkers8.codebook import encode_fen, pad_sequence, encode_move, NUM_MOVES
from checkers8.notation import fen_to_board
from checkers8.model import CheckersTransformer


class CheckersDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        fen = item['fen']
        move_idx = item['move_idx']
        token_ids = pad_sequence(encode_fen(fen))

        board = fen_to_board(fen)
        legal = [False] * NUM_MOVES
        for move in board.legal_moves():
            origin, steps = move
            final = steps[-1]
            legal[encode_move(origin, final)] = True

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(legal, dtype=torch.bool),
        )


def save_cpu_state(model, path):
    # Always persist CPU tensors so the Flask app loads cleanly on a CPU-only
    # machine even when the weights were trained on a GPU (e.g. Colab).
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def train(epochs=200, lr=1e-3, batch_size=512, data_path='checkers8/data.json',
          out_path='checkers8/model.pt', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    with open(data_path) as f:
        data = json.load(f)

    print(f'Training on {len(data)} positions')
    dataset = CheckersDataset(data)
    pin = device == 'cuda'
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=pin)

    model = CheckersTransformer().to(device)
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
        if acc > best_acc:
            best_acc = acc
            save_cpu_state(model, out_path)

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / total
            print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}', flush=True)

    save_cpu_state(model, out_path)
    print(f'Model saved to {out_path}. Best accuracy: {best_acc:.4f}')
    return model


if __name__ == '__main__':
    train()
