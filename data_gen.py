import json
from game import Board, X, minimax
from notation import board_to_fen


def generate_all_positions():
    dataset = []
    seen = set()

    def walk(board):
        if board.is_terminal():
            return
        fen = board_to_fen(board)
        if fen in seen:
            return
        seen.add(fen)
        _, best_move = minimax(board)
        dataset.append({'fen': fen, 'move': best_move})
        for move in board.legal_moves():
            walk(board.make_move(move))

    walk(Board())
    return dataset


if __name__ == '__main__':
    data = generate_all_positions()
    with open('data.json', 'w') as f:
        json.dump(data, f)
    print(f'Generated {len(data)} training examples')
