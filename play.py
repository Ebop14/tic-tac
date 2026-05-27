import torch
from game import Board, X, O, minimax
from notation import board_to_fen
from codebook import encode_fen, pad_sequence, NUM_MOVES
from model import TicTacToeTransformer


def model_pick_move(model, board):
    fen = board_to_fen(board)
    token_ids = pad_sequence(encode_fen(fen))
    x = torch.tensor([token_ids], dtype=torch.long)
    legal = [False] * NUM_MOVES
    for m in board.legal_moves():
        legal[m] = True
    mask = torch.tensor([legal], dtype=torch.bool)
    with torch.no_grad():
        logits = model(x, legal_mask=mask)
    return logits.argmax(dim=1).item()


def play_game(model, model_plays_as):
    board = Board()
    while not board.is_terminal():
        if board.current_player == model_plays_as:
            move = model_pick_move(model, board)
        else:
            _, move = minimax(board)
        board = board.make_move(move)
    return board.result()


def evaluate(model, num_games=100):
    results = {'X_wins': 0, 'O_wins': 0, 'draws': 0}

    for side in [X, O]:
        for _ in range(num_games):
            r = play_game(model, side)
            if r == 1:
                results['X_wins'] += 1
            elif r == -1:
                results['O_wins'] += 1
            else:
                results['draws'] += 1

    total = num_games * 2
    label = {X: 'X', O: 'O'}
    print(f'Results over {total} games ({num_games} as X, {num_games} as O):')
    print(f"  Model wins as X: {results['X_wins']} / {num_games}")
    # When model plays O, "O_wins" is model winning
    # Need to track differently - let me recount
    return results


def evaluate_detailed(model, num_games=100):
    model_wins = 0
    model_losses = 0
    draws = 0

    for side in [X, O]:
        side_wins = 0
        side_losses = 0
        side_draws = 0
        for _ in range(num_games):
            r = play_game(model, side)
            if (side == X and r == 1) or (side == O and r == -1):
                side_wins += 1
                model_wins += 1
            elif r == 0:
                side_draws += 1
                draws += 1
            else:
                side_losses += 1
                model_losses += 1
        label = 'X' if side == X else 'O'
        print(f'  As {label}: {side_wins}W / {side_draws}D / {side_losses}L')

    total = num_games * 2
    print(f'\nTotal ({total} games): {model_wins}W / {draws}D / {model_losses}L')


if __name__ == '__main__':
    model = TicTacToeTransformer()
    model.load_state_dict(torch.load('model.pt', weights_only=True))
    model.eval()
    print('Trained model vs Minimax:')
    evaluate_detailed(model)
