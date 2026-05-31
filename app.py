import math
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from game import Board, X, O
from notation import board_to_fen
from codebook import encode_fen, pad_sequence, NUM_MOVES
from model import TicTacToeTransformer

from checkers.game import (
    Board as CBoard, BLACK as C_BLACK, WHITE as C_WHITE,
    EMPTY as C_EMPTY, BLACK_MAN, WHITE_MAN, BLACK_KING, WHITE_KING,
    owner as c_owner, NUM_SQUARES as C_NUM_SQUARES,
    SQUARE_TO_RC, RC_TO_SQUARE,
)
from checkers.notation import board_to_fen as c_board_to_fen, fen_to_board as c_fen_to_board
from checkers.codebook import (
    encode_fen as c_encode_fen, pad_sequence as c_pad_sequence,
    encode_move as c_encode_move, NUM_MOVES as C_NUM_MOVES,
)
from checkers.model import CheckersTransformer

from checkers8.game import (
    SQUARE_TO_RC as C8_SQUARE_TO_RC, RC_TO_SQUARE as C8_RC_TO_SQUARE,
)
from checkers8.notation import board_to_fen as c8_board_to_fen, fen_to_board as c8_fen_to_board
from checkers8.codebook import (
    encode_fen as c8_encode_fen, pad_sequence as c8_pad_sequence,
    encode_move as c8_encode_move, NUM_MOVES as C8_NUM_MOVES,
)
from checkers8.model import CheckersTransformer as Checkers8Transformer

from checkers8free.game import (
    SQUARE_TO_RC as C8F_SQUARE_TO_RC, RC_TO_SQUARE as C8F_RC_TO_SQUARE,
)
from checkers8free.notation import board_to_fen as c8f_board_to_fen, fen_to_board as c8f_fen_to_board
from checkers8free.codebook import (
    encode_fen as c8f_encode_fen, pad_sequence as c8f_pad_sequence,
    encode_move as c8f_encode_move, NUM_MOVES as C8F_NUM_MOVES,
)
from checkers8free.model import CheckersTransformer as Checkers8FreeTransformer

app = Flask(__name__)

ttt_model = TicTacToeTransformer()
ttt_model.load_state_dict(torch.load('model.pt', weights_only=True))
ttt_model.eval()

checkers_model = CheckersTransformer()
checkers_model.load_state_dict(torch.load('checkers/model.pt', weights_only=True))
checkers_model.eval()

checkers8_model = Checkers8Transformer()
checkers8_model.load_state_dict(torch.load('checkers8/model.pt', weights_only=True))
checkers8_model.eval()

checkers8free_model = Checkers8FreeTransformer()
checkers8free_model.load_state_dict(torch.load('checkers8free/model.pt', weights_only=True))
checkers8free_model.eval()


def sanitize_logits(logits):
    return [None if math.isinf(v) else round(v, 3) for v in logits]


# --- Move sampling (temperature + top-k) ---
#
# Difficulty presets map to (temperature, top_k). Higher temperature spreads
# probability across moves (more human, more mistakes); top_k caps how many of
# the best moves are even considered, so the model never plays an absurd blunder.
# "sharp" with top_k=1 reproduces the old deterministic argmax (unbeatable).
DIFFICULTY = {
    'chill': (1.6, 5),
    'balanced': (0.9, 3),
    'sharp': (1.0, 1),
}


def resolve_sampling(data):
    name = str(data.get('difficulty', 'balanced')).lower()
    return DIFFICULTY.get(name, DIFFICULTY['balanced'])


def sample_index(logits_row, temperature=1.0, top_k=1):
    """Pick a move index from a logits row (illegal moves are -inf).

    Falls back to argmax when there's nothing to randomize over. Legal logits
    are first scaled to unit spread (divided by their std) so a given
    temperature feels the same regardless of how peaked the model is for this
    position or game — minimax-trained models otherwise produce enormous logit
    gaps that swamp any reasonable temperature.
    """
    logits = logits_row.clone()
    finite = torch.isfinite(logits)
    n_legal = int(finite.sum().item())
    if n_legal <= 1 or top_k <= 1 or temperature <= 0:
        return int(torch.argmax(logits).item())
    scale = logits[finite].std().item()
    if scale < 1e-6:
        return int(torch.argmax(logits).item())
    logits = logits / scale
    k = min(top_k, n_legal)
    top_vals, top_idx = torch.topk(logits, k)
    filtered = torch.full_like(logits, float('-inf'))
    filtered[top_idx] = top_vals
    probs = F.softmax(filtered / temperature, dim=0)
    return int(torch.multinomial(probs, 1).item())


# --- Tic-Tac-Toe ---

def ttt_move_with_details(board, temperature=1.0, top_k=1):
    fen = board_to_fen(board)
    token_ids = pad_sequence(encode_fen(fen))
    x = torch.tensor([token_ids], dtype=torch.long)
    legal = [False] * NUM_MOVES
    for m in board.legal_moves():
        legal[m] = True
    mask = torch.tensor([legal], dtype=torch.bool)
    with torch.no_grad():
        logits = ttt_model(x, legal_mask=mask)
    probs = F.softmax(logits, dim=1)
    move = sample_index(logits[0], temperature, top_k)
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

    temperature, top_k = resolve_sampling(data)
    move, logits, probs = ttt_move_with_details(board, temperature, top_k)
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

    model_move, logits, probs = ttt_move_with_details(after_human)
    return jsonify({
        'previewMove': preview_move_idx,
        'isTerminal': False,
        'winner': None,
        'logits': sanitize_logits(logits),
        'probs': [round(v, 4) for v in probs],
        'modelMove': model_move,
    })


# --- Checkers ---

def checkers_infer(board, temperature=1.0, top_k=1):
    fen = c_board_to_fen(board)
    tokens = c_pad_sequence(c_encode_fen(fen))
    x = torch.tensor([tokens], dtype=torch.long)
    legal = [False] * C_NUM_MOVES
    move_map = {}
    for move in board.legal_moves():
        origin, steps = move
        mi = c_encode_move(origin, steps[-1])
        legal[mi] = True
        move_map[mi] = move
    mask = torch.tensor([legal], dtype=torch.bool)
    with torch.no_grad():
        logits = checkers_model(x, legal_mask=mask)
    probs = F.softmax(logits, dim=1)
    pred_idx = sample_index(logits[0], temperature, top_k)
    return move_map[pred_idx], logits[0].tolist(), probs[0].tolist(), move_map


def checkers_legal_json(board):
    moves = []
    if board.is_terminal():
        return moves
    for move in board.legal_moves():
        origin, steps = move
        moves.append({'origin': origin, 'dest': steps[-1], 'steps': steps})
    return moves


def checkers_captured(move):
    origin, steps = move
    captured = []
    prev = origin
    for step in steps:
        pr, pc = SQUARE_TO_RC[prev]
        sr, sc = SQUARE_TO_RC[step]
        if abs(sr - pr) == 2:
            mid_sq = RC_TO_SQUARE[((pr + sr) // 2, (pc + sc) // 2)]
            captured.append(mid_sq)
        prev = step
    return captured


def checkers_move_probs(board, logits_list, probs_list):
    is_jump = len(board._jump_moves()) > 0
    sep = 'x' if is_jump else '-'
    result = []
    for move in board.legal_moves():
        origin, steps = move
        mi = c_encode_move(origin, steps[-1])
        pdn = sep.join(str(s) for s in [origin] + steps)
        after = board.make_move(move)
        result.append({
            'origin': origin,
            'dest': steps[-1],
            'pdn': pdn,
            'fen': c_board_to_fen(after),
            'prob': round(probs_list[mi], 4),
            'logit': sanitize_logits([logits_list[mi]])[0],
        })
    result.sort(key=lambda x: x['prob'], reverse=True)
    return result


@app.route('/checkers')
def checkers_index():
    return render_template('checkers.html')


@app.route('/checkers/api/state', methods=['POST'])
def checkers_state():
    data = request.get_json()
    fen = data['fen']
    board = c_fen_to_board(fen)
    return jsonify({
        'squares': board.squares,
        'currentPlayer': board.current_player,
        'legalMoves': checkers_legal_json(board),
        'isTerminal': board.is_terminal(),
        'result': board.result() if board.is_terminal() else None,
    })


@app.route('/checkers/api/move', methods=['POST'])
def checkers_move():
    data = request.get_json()
    fen = data['fen']
    origin = data['origin']
    dest = data['dest']

    board = c_fen_to_board(fen)
    target = None
    for move in board.legal_moves():
        o, steps = move
        if o == origin and steps[-1] == dest:
            target = move
            break

    if target is None:
        return jsonify({'error': 'Invalid move'}), 400

    after_user = board.make_move(target)
    user_fen = c_board_to_fen(after_user)

    if after_user.is_terminal():
        return jsonify({
            'userFen': user_fen,
            'fen': user_fen,
            'squares': after_user.squares,
            'isTerminal': True,
            'result': after_user.result(),
            'modelResponse': None,
            'legalMoves': [],
        })

    temperature, top_k = resolve_sampling(data)
    model_move, logits, probs, _ = checkers_infer(after_user, temperature, top_k)
    move_probs = checkers_move_probs(after_user, logits, probs)
    after_model = after_user.make_move(model_move)
    model_fen = c_board_to_fen(after_model)

    m_origin, m_steps = model_move

    return jsonify({
        'userFen': user_fen,
        'userSquares': after_user.squares,
        'fen': model_fen,
        'squares': after_model.squares,
        'isTerminal': after_model.is_terminal(),
        'result': after_model.result() if after_model.is_terminal() else None,
        'modelResponse': {
            'origin': m_origin,
            'dest': m_steps[-1],
            'steps': m_steps,
            'moveProbs': move_probs,
        },
        'legalMoves': checkers_legal_json(after_model),
    })


@app.route('/checkers/api/preview', methods=['POST'])
def checkers_preview():
    data = request.get_json()
    fen = data['fen']
    origin = data['origin']
    dest = data['dest']

    board = c_fen_to_board(fen)
    target = None
    for move in board.legal_moves():
        o, steps = move
        if o == origin and steps[-1] == dest:
            target = move
            break

    if target is None:
        return jsonify({'error': 'Invalid preview'}), 400

    captured = checkers_captured(target)
    after_user = board.make_move(target)
    user_fen = c_board_to_fen(after_user)

    if after_user.is_terminal():
        return jsonify({
            'userFen': user_fen,
            'captured': captured,
            'isTerminal': True,
            'result': after_user.result(),
            'modelResponse': None,
        })

    model_move, logits, probs, _ = checkers_infer(after_user)
    move_probs = checkers_move_probs(after_user, logits, probs)
    m_origin, m_steps = model_move

    return jsonify({
        'userFen': user_fen,
        'captured': captured,
        'isTerminal': False,
        'modelResponse': {
            'origin': m_origin,
            'dest': m_steps[-1],
            'moveProbs': move_probs,
        },
    })


# --- Checkers (8x8, full size) ---

def checkers8_infer(board, temperature=1.0, top_k=1):
    fen = c8_board_to_fen(board)
    tokens = c8_pad_sequence(c8_encode_fen(fen))
    x = torch.tensor([tokens], dtype=torch.long)
    legal = [False] * C8_NUM_MOVES
    move_map = {}
    for move in board.legal_moves():
        origin, steps = move
        mi = c8_encode_move(origin, steps[-1])
        legal[mi] = True
        move_map[mi] = move
    mask = torch.tensor([legal], dtype=torch.bool)
    with torch.no_grad():
        logits = checkers8_model(x, legal_mask=mask)
    probs = F.softmax(logits, dim=1)
    pred_idx = sample_index(logits[0], temperature, top_k)
    return move_map[pred_idx], logits[0].tolist(), probs[0].tolist(), move_map


def checkers8_legal_json(board):
    moves = []
    if board.is_terminal():
        return moves
    for move in board.legal_moves():
        origin, steps = move
        moves.append({'origin': origin, 'dest': steps[-1], 'steps': steps})
    return moves


def checkers8_captured(move):
    origin, steps = move
    captured = []
    prev = origin
    for step in steps:
        pr, pc = C8_SQUARE_TO_RC[prev]
        sr, sc = C8_SQUARE_TO_RC[step]
        if abs(sr - pr) == 2:
            mid_sq = C8_RC_TO_SQUARE[((pr + sr) // 2, (pc + sc) // 2)]
            captured.append(mid_sq)
        prev = step
    return captured


def checkers8_move_probs(board, logits_list, probs_list):
    is_jump = len(board._jump_moves()) > 0
    sep = 'x' if is_jump else '-'
    result = []
    for move in board.legal_moves():
        origin, steps = move
        mi = c8_encode_move(origin, steps[-1])
        pdn = sep.join(str(s) for s in [origin] + steps)
        after = board.make_move(move)
        result.append({
            'origin': origin,
            'dest': steps[-1],
            'pdn': pdn,
            'fen': c8_board_to_fen(after),
            'prob': round(probs_list[mi], 4),
            'logit': sanitize_logits([logits_list[mi]])[0],
        })
    result.sort(key=lambda x: x['prob'], reverse=True)
    return result


@app.route('/checkers8')
def checkers8_index():
    return render_template('checkers8.html')


@app.route('/checkers8/api/state', methods=['POST'])
def checkers8_state():
    data = request.get_json()
    fen = data['fen']
    board = c8_fen_to_board(fen)
    return jsonify({
        'squares': board.squares,
        'currentPlayer': board.current_player,
        'legalMoves': checkers8_legal_json(board),
        'isTerminal': board.is_terminal(),
        'result': board.result() if board.is_terminal() else None,
    })


@app.route('/checkers8/api/move', methods=['POST'])
def checkers8_move():
    data = request.get_json()
    fen = data['fen']
    origin = data['origin']
    dest = data['dest']

    board = c8_fen_to_board(fen)
    target = None
    for move in board.legal_moves():
        o, steps = move
        if o == origin and steps[-1] == dest:
            target = move
            break

    if target is None:
        return jsonify({'error': 'Invalid move'}), 400

    after_user = board.make_move(target)
    user_fen = c8_board_to_fen(after_user)

    if after_user.is_terminal():
        return jsonify({
            'userFen': user_fen,
            'fen': user_fen,
            'squares': after_user.squares,
            'isTerminal': True,
            'result': after_user.result(),
            'modelResponse': None,
            'legalMoves': [],
        })

    temperature, top_k = resolve_sampling(data)
    model_move, logits, probs, _ = checkers8_infer(after_user, temperature, top_k)
    move_probs = checkers8_move_probs(after_user, logits, probs)
    after_model = after_user.make_move(model_move)
    model_fen = c8_board_to_fen(after_model)

    m_origin, m_steps = model_move

    return jsonify({
        'userFen': user_fen,
        'userSquares': after_user.squares,
        'fen': model_fen,
        'squares': after_model.squares,
        'isTerminal': after_model.is_terminal(),
        'result': after_model.result() if after_model.is_terminal() else None,
        'modelResponse': {
            'origin': m_origin,
            'dest': m_steps[-1],
            'steps': m_steps,
            'moveProbs': move_probs,
        },
        'legalMoves': checkers8_legal_json(after_model),
    })


@app.route('/checkers8/api/preview', methods=['POST'])
def checkers8_preview():
    data = request.get_json()
    fen = data['fen']
    origin = data['origin']
    dest = data['dest']

    board = c8_fen_to_board(fen)
    target = None
    for move in board.legal_moves():
        o, steps = move
        if o == origin and steps[-1] == dest:
            target = move
            break

    if target is None:
        return jsonify({'error': 'Invalid preview'}), 400

    captured = checkers8_captured(target)
    after_user = board.make_move(target)
    user_fen = c8_board_to_fen(after_user)

    if after_user.is_terminal():
        return jsonify({
            'userFen': user_fen,
            'captured': captured,
            'isTerminal': True,
            'result': after_user.result(),
            'modelResponse': None,
        })

    model_move, logits, probs, _ = checkers8_infer(after_user)
    move_probs = checkers8_move_probs(after_user, logits, probs)
    m_origin, m_steps = model_move

    return jsonify({
        'userFen': user_fen,
        'captured': captured,
        'isTerminal': False,
        'modelResponse': {
            'origin': m_origin,
            'dest': m_steps[-1],
            'moveProbs': move_probs,
        },
    })


# --- Checkers (8x8, captures optional) ---

def checkers8free_infer(board, temperature=1.0, top_k=1):
    fen = c8f_board_to_fen(board)
    tokens = c8f_pad_sequence(c8f_encode_fen(fen))
    x = torch.tensor([tokens], dtype=torch.long)
    legal = [False] * C8F_NUM_MOVES
    move_map = {}
    for move in board.legal_moves():
        origin, steps = move
        mi = c8f_encode_move(origin, steps[-1])
        legal[mi] = True
        move_map[mi] = move
    mask = torch.tensor([legal], dtype=torch.bool)
    with torch.no_grad():
        logits = checkers8free_model(x, legal_mask=mask)
    probs = F.softmax(logits, dim=1)
    pred_idx = sample_index(logits[0], temperature, top_k)
    return move_map[pred_idx], logits[0].tolist(), probs[0].tolist(), move_map


def checkers8free_legal_json(board):
    moves = []
    if board.is_terminal():
        return moves
    for move in board.legal_moves():
        origin, steps = move
        moves.append({'origin': origin, 'dest': steps[-1], 'steps': steps})
    return moves


def checkers8free_captured(move):
    origin, steps = move
    captured = []
    prev = origin
    for step in steps:
        pr, pc = C8F_SQUARE_TO_RC[prev]
        sr, sc = C8F_SQUARE_TO_RC[step]
        if abs(sr - pr) == 2:
            mid_sq = C8F_RC_TO_SQUARE[((pr + sr) // 2, (pc + sc) // 2)]
            captured.append(mid_sq)
        prev = step
    return captured


def checkers8free_move_probs(board, logits_list, probs_list):
    result = []
    for move in board.legal_moves():
        origin, steps = move
        mi = c8f_encode_move(origin, steps[-1])
        # Captures are optional here, so a single position can mix jumps and
        # quiet moves; pick the separator per-move from its own length.
        sep = 'x' if len(steps) > 1 or checkers8free_captured(move) else '-'
        pdn = sep.join(str(s) for s in [origin] + steps)
        after = board.make_move(move)
        result.append({
            'origin': origin,
            'dest': steps[-1],
            'pdn': pdn,
            'fen': c8f_board_to_fen(after),
            'prob': round(probs_list[mi], 4),
            'logit': sanitize_logits([logits_list[mi]])[0],
        })
    result.sort(key=lambda x: x['prob'], reverse=True)
    return result


@app.route('/checkers8free')
def checkers8free_index():
    return render_template('checkers8free.html')


@app.route('/checkers8free/api/state', methods=['POST'])
def checkers8free_state():
    data = request.get_json()
    fen = data['fen']
    board = c8f_fen_to_board(fen)
    return jsonify({
        'squares': board.squares,
        'currentPlayer': board.current_player,
        'legalMoves': checkers8free_legal_json(board),
        'isTerminal': board.is_terminal(),
        'result': board.result() if board.is_terminal() else None,
    })


@app.route('/checkers8free/api/move', methods=['POST'])
def checkers8free_move():
    data = request.get_json()
    fen = data['fen']
    origin = data['origin']
    dest = data['dest']

    board = c8f_fen_to_board(fen)
    target = None
    for move in board.legal_moves():
        o, steps = move
        if o == origin and steps[-1] == dest:
            target = move
            break

    if target is None:
        return jsonify({'error': 'Invalid move'}), 400

    after_user = board.make_move(target)
    user_fen = c8f_board_to_fen(after_user)

    if after_user.is_terminal():
        return jsonify({
            'userFen': user_fen,
            'fen': user_fen,
            'squares': after_user.squares,
            'isTerminal': True,
            'result': after_user.result(),
            'modelResponse': None,
            'legalMoves': [],
        })

    temperature, top_k = resolve_sampling(data)
    model_move, logits, probs, _ = checkers8free_infer(after_user, temperature, top_k)
    move_probs = checkers8free_move_probs(after_user, logits, probs)
    after_model = after_user.make_move(model_move)
    model_fen = c8f_board_to_fen(after_model)

    m_origin, m_steps = model_move

    return jsonify({
        'userFen': user_fen,
        'userSquares': after_user.squares,
        'fen': model_fen,
        'squares': after_model.squares,
        'isTerminal': after_model.is_terminal(),
        'result': after_model.result() if after_model.is_terminal() else None,
        'modelResponse': {
            'origin': m_origin,
            'dest': m_steps[-1],
            'steps': m_steps,
            'moveProbs': move_probs,
        },
        'legalMoves': checkers8free_legal_json(after_model),
    })


@app.route('/checkers8free/api/preview', methods=['POST'])
def checkers8free_preview():
    data = request.get_json()
    fen = data['fen']
    origin = data['origin']
    dest = data['dest']

    board = c8f_fen_to_board(fen)
    target = None
    for move in board.legal_moves():
        o, steps = move
        if o == origin and steps[-1] == dest:
            target = move
            break

    if target is None:
        return jsonify({'error': 'Invalid preview'}), 400

    captured = checkers8free_captured(target)
    after_user = board.make_move(target)
    user_fen = c8f_board_to_fen(after_user)

    if after_user.is_terminal():
        return jsonify({
            'userFen': user_fen,
            'captured': captured,
            'isTerminal': True,
            'result': after_user.result(),
            'modelResponse': None,
        })

    model_move, logits, probs, _ = checkers8free_infer(after_user)
    move_probs = checkers8free_move_probs(after_user, logits, probs)
    m_origin, m_steps = model_move

    return jsonify({
        'userFen': user_fen,
        'captured': captured,
        'isTerminal': False,
        'modelResponse': {
            'origin': m_origin,
            'dest': m_steps[-1],
            'moveProbs': move_probs,
        },
    })


if __name__ == '__main__':
    app.run(debug=True, port=5050)
