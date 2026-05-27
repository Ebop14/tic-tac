import json
from checkers.game import Board, minimax
from checkers.notation import board_to_fen
from checkers.codebook import encode_move


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
        val, best_move = minimax(board)
        if best_move is None:
            return
        origin, steps = best_move
        final_dest = steps[-1]
        move_idx = encode_move(origin, final_dest)
        dataset.append({
            'fen': fen,
            'move_idx': move_idx,
            'origin': origin,
            'dest': final_dest,
            'val': val,
        })
        for move in board.legal_moves():
            walk(board.make_move(move))

    walk(Board())
    return dataset


if __name__ == '__main__':
    data = generate_all_positions()
    with open('checkers/data.json', 'w') as f:
        json.dump(data, f)
    print(f'Generated {len(data)} training examples')
