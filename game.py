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
