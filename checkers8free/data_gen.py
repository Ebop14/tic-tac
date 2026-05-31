import json
import random
import sys
import time

from checkers8free.game import Board
from checkers8free.notation import board_to_fen
from checkers8free.codebook import encode_move
from checkers8free.game import search


def play_game(depth, epsilon, max_moves, rng):
    """Self-play one game. At each position record the depth-limited best move
    as the training label, but actually play an epsilon-greedy move so the
    sampled positions stay diverse."""
    board = Board()
    samples = []
    for _ in range(max_moves):
        if board.is_terminal():
            break
        val, best = search(board, depth)
        if best is None:
            break
        origin, steps = best
        final = steps[-1]
        samples.append({
            'fen': board_to_fen(board),
            'move_idx': encode_move(origin, final),
            'origin': origin,
            'dest': final,
            'val': val,
        })
        legal = board.legal_moves()
        if rng.random() < epsilon and len(legal) > 1:
            move = rng.choice(legal)
        else:
            move = best
        board = board.make_move(move)
    return samples


def generate_dataset(num_games=300, depth=5, epsilon=0.25, max_moves=90, seed=0):
    rng = random.Random(seed)
    by_fen = {}
    start = time.time()
    for g in range(num_games):
        for sample in play_game(depth, epsilon, max_moves, rng):
            by_fen.setdefault(sample['fen'], sample)
        if (g + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f'  game {g + 1}/{num_games} | {len(by_fen)} unique positions '
                  f'| {elapsed:.0f}s', flush=True)
    return list(by_fen.values())


if __name__ == '__main__':
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    data = generate_dataset(num_games=games, depth=depth)
    with open('checkers8free/data.json', 'w') as f:
        json.dump(data, f)
    print(f'Generated {len(data)} training examples (games={games}, depth={depth})')
