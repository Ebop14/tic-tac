import json
from game import Board, shaped_move_values
from notation import board_to_fen

NUM_MOVES = 9


def generate_all_positions():
    """One example per reachable non-terminal position.

    Each example carries a length-9 vector of depth-discounted minimax values
    (from the mover's perspective) for legal moves; illegal squares are 0.0 and
    ignored at train time via the legal mask.
    """
    dataset = []
    seen = set()

    def walk(board):
        if board.is_terminal():
            return
        fen = board_to_fen(board)
        if fen in seen:
            return
        seen.add(fen)
        values = [0.0] * NUM_MOVES
        for move, val in shaped_move_values(board).items():
            values[move] = val
        dataset.append({'fen': fen, 'values': values})
        for move in board.legal_moves():
            walk(board.make_move(move))

    walk(Board())
    return dataset


if __name__ == '__main__':
    data = generate_all_positions()
    with open('data.json', 'w') as f:
        json.dump(data, f)
    print(f'Generated {len(data)} training examples')
