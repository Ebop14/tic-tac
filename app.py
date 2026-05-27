import math
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from game import Board, X, O
from notation import board_to_fen
from codebook import encode_fen, pad_sequence, NUM_MOVES
from model import TicTacToeTransformer

app = Flask(__name__)

model = TicTacToeTransformer()
model.load_state_dict(torch.load('model.pt', weights_only=True))
model.eval()


def model_move_with_details(model, board):
    fen = board_to_fen(board)
    token_ids = pad_sequence(encode_fen(fen))
    x = torch.tensor([token_ids], dtype=torch.long)
    legal = [False] * NUM_MOVES
    for m in board.legal_moves():
        legal[m] = True
    mask = torch.tensor([legal], dtype=torch.bool)
    with torch.no_grad():
        logits = model(x, legal_mask=mask)
    probs = F.softmax(logits, dim=1)
    move = logits.argmax(dim=1).item()
    return move, logits[0].tolist(), probs[0].tolist()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/move', methods=['POST'])
def get_move():
    data = request.get_json()
    squares = data['squares']
    current_player = data['currentPlayer']
    board = Board(squares, current_player)

    if board.is_terminal():
        return jsonify({'error': 'Game is already over'}), 400

    move, logits, probs = model_move_with_details(model, board)
    new_board = board.make_move(move)

    winner = new_board.check_winner()
    winner_label = None
    if winner == X:
        winner_label = 'X'
    elif winner == O:
        winner_label = 'O'

    return jsonify({
        'move': move,
        'squares': new_board.squares,
        'currentPlayer': new_board.current_player,
        'isTerminal': new_board.is_terminal(),
        'winner': winner_label,
        'logits': sanitize_logits(logits),
        'probs': [round(v, 4) for v in probs],
    })


def sanitize_logits(logits):
    return [None if math.isinf(v) else round(v, 3) for v in logits]


@app.route('/api/preview', methods=['POST'])
def preview_move():
    data = request.get_json()
    squares = data['squares']
    current_player = data['currentPlayer']
    preview_move_idx = data['move']
    board = Board(squares, current_player)

    if board.is_terminal() or squares[preview_move_idx] != 0:
        return jsonify({'error': 'Invalid preview'}), 400

    after_human = board.make_move(preview_move_idx)

    if after_human.is_terminal():
        winner = after_human.check_winner()
        winner_label = 'X' if winner == X else ('O' if winner == O else None)
        return jsonify({
            'previewMove': preview_move_idx,
            'isTerminal': True,
            'winner': winner_label,
            'logits': [None] * 9,
            'probs': [0.0] * 9,
            'modelMove': None,
        })

    model_move, logits, probs = model_move_with_details(model, after_human)
    return jsonify({
        'previewMove': preview_move_idx,
        'isTerminal': False,
        'winner': None,
        'logits': sanitize_logits(logits),
        'probs': [round(v, 4) for v in probs],
        'modelMove': model_move,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5050)
