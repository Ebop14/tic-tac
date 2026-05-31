EMPTY = 0
X = 1
O = 2

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


class Board:
    def __init__(self, squares=None, current_player=X):
        self.squares = list(squares) if squares else [EMPTY] * 9
        self.current_player = current_player

    def copy(self):
        return Board(self.squares, self.current_player)

    def legal_moves(self):
        return [i for i in range(9) if self.squares[i] == EMPTY]

    def make_move(self, pos):
        b = self.copy()
        b.squares[pos] = b.current_player
        b.current_player = O if b.current_player == X else X
        return b

    def check_winner(self):
        for a, b, c in WIN_LINES:
            if self.squares[a] != EMPTY and self.squares[a] == self.squares[b] == self.squares[c]:
                return self.squares[a]
        return None

    def is_terminal(self):
        return self.check_winner() is not None or len(self.legal_moves()) == 0

    def result(self):
        w = self.check_winner()
        if w == X:
            return 1
        if w == O:
            return -1
        return 0


def minimax(board, alpha=-2, beta=2):
    if board.is_terminal():
        return board.result(), None

    if board.current_player == X:
        best_val = -2
        best_move = None
        for move in board.legal_moves():
            val, _ = minimax(board.make_move(move), alpha, beta)
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best_val, best_move
    else:
        best_val = 2
        best_move = None
        for move in board.legal_moves():
            val, _ = minimax(board.make_move(move), alpha, beta)
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, val)
            if alpha >= beta:
                break
        return best_val, best_move


# --- Depth-discounted minimax values ---
#
# Absolute convention (X maximizes, O minimizes). Terminal nodes return the
# raw result (+1 / 0 / -1); internal nodes back up gamma * best_child so the
# magnitude decays with distance to the end. A win-in-3 (gamma**2) therefore
# scores higher than a win-in-7, and a loss-in-5 is less bad than a loss-in-1.
# No alpha-beta here: discounting breaks the simple ±bound pruning, and the
# tic-tac-toe tree is small enough to search exhaustively (memoized).
GAMMA = 0.9
_value_cache = {}


def minimax_value(board, gamma=GAMMA):
    if board.is_terminal():
        return board.result()
    key = (tuple(board.squares), board.current_player)
    cached = _value_cache.get(key)
    if cached is not None:
        return cached
    child_vals = [minimax_value(board.make_move(m), gamma) for m in board.legal_moves()]
    backed = max(child_vals) if board.current_player == X else min(child_vals)
    val = gamma * backed
    _value_cache[key] = val
    return val


def move_values(board, gamma=GAMMA):
    """Per-move values from the perspective of the player to move.

    Higher is always better for whoever is on the move, regardless of side.
    """
    sign = 1 if board.current_player == X else -1
    return {m: sign * minimax_value(board.make_move(m), gamma)
            for m in board.legal_moves()}
