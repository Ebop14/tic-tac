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
        move = item['move']
        token_ids = pad_sequence(encode_fen(fen))
        board = fen_to_board(fen)
        legal = [False] * NUM_MOVES
        for m in board.legal_moves():
            legal[m] = True
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(move, dtype=torch.long),
            torch.tensor(legal, dtype=torch.bool),
        )


def train(epochs=800, lr=1e-3, batch_size=64):
    with open('data.json') as f:
        data = json.load(f)

    dataset = TTTDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TicTacToeTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for x, y, mask in loader:
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
            torch.save(model.state_dict(), 'model.pt')

        if epoch % 25 == 0 or epoch == 1:
            avg_loss = total_loss / total
            print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}')

    torch.save(model.state_dict(), 'model.pt')
    print('Model saved to model.pt')
    return model


if __name__ == '__main__':
    train()
