import json
import math
import os
import random
import sys
import time

from checkers8free.game import Board, search, BLACK
from checkers8free.notation import board_to_fen, fen_to_board
from checkers8free.codebook import encode_move

# Heuristic search values are unbounded material scores (a man is ~100, a king
# ~175, terminal wins ~+/-WIN). tanh(value / VALUE_SCALE) squashes them into the
# model's [-1, 1] value head: ~1 man -> 0.32, 2 men -> 0.58, 3 men -> 0.76, and
# terminal wins saturate to +/-1. Difficulty tolerances downstream are in these
# squashed units.
VALUE_SCALE = 300.0


def squash(v):
    return math.tanh(v / VALUE_SCALE)


def position_values(board, depth):
    """Per-move squashed values from the perspective of the player to move.

    For each legal move we look one ply past it (search at depth-1), which is
    exactly the value the depth-`depth` teacher would back up for that move.
    Sign-flipped for WHITE so higher is always better for whoever is on move.
    """
    sign = 1 if board.current_player == BLACK else -1
    out = []
    for move in board.legal_moves():
        origin, steps = move
        v, _ = search(board.make_move(move), depth - 1)
        out.append([encode_move(origin, steps[-1]), squash(sign * v)])
    return out


# --- position sourcing via self-play (the same teacher as before) ---

def play_game(depth, epsilon, max_moves, rng):
    board = Board()
    fens = []
    for _ in range(max_moves):
        if board.is_terminal():
            break
        _, best = search(board, depth)
        if best is None:
            break
        fens.append(board_to_fen(board))
        legal = board.legal_moves()
        if rng.random() < epsilon and len(legal) > 1:
            move = rng.choice(legal)
        else:
            move = best
        board = board.make_move(move)
    return fens


def collect_fens(num_games, depth, epsilon, max_moves, seed):
    rng = random.Random(seed)
    seen = set()
    for g in range(num_games):
        for fen in play_game(depth, epsilon, max_moves, rng):
            seen.add(fen)
        if (g + 1) % 10 == 0:
            print(f'  game {g + 1}/{num_games} | {len(seen)} unique positions', flush=True)
    return list(seen)


def build_value_dataset(fens, depth):
    dataset = []
    start = time.time()
    for i, fen in enumerate(fens):
        board = fen_to_board(fen)
        if board.is_terminal():
            continue
        values = position_values(board, depth)
        if not values:
            continue
        dataset.append({'fen': fen, 'values': values})
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start
            print(f'  valued {i + 1}/{len(fens)} positions | {elapsed:.0f}s', flush=True)
    return dataset


if __name__ == '__main__':
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    existing = 'checkers8free/data.json'
    # Reuse the teacher's already-sampled positions when present (deterministic,
    # same distribution as the original classifier dataset); otherwise self-play.
    fens = None
    if os.path.exists(existing):
        with open(existing) as f:
            prev = json.load(f)
        if prev and 'fen' in prev[0]:
            fens = [item['fen'] for item in prev]
            print(f'Reusing {len(fens)} positions from {existing}')
    if fens is None:
        print('No existing positions found; generating via self-play')
        fens = collect_fens(num_games=300, depth=depth, epsilon=0.25, max_moves=90, seed=0)

    data = build_value_dataset(fens, depth)
    with open('checkers8free/data.json', 'w') as f:
        json.dump(data, f)
    print(f'Generated {len(data)} value-labeled examples (depth={depth}, scale={VALUE_SCALE})')
