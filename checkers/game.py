EMPTY = 0
BLACK_MAN = 1
WHITE_MAN = 2
BLACK_KING = 3
WHITE_KING = 4

BLACK = 1
WHITE = 2

NUM_SQUARES = 8

# Square numbering for 4x4 board (dark squares only):
#   .  1  .  2
#   3  .  4  .
#   .  5  .  6
#   7  .  8  .
SQUARE_TO_RC = {
    1: (0, 1), 2: (0, 3),
    3: (1, 0), 4: (1, 2),
    5: (2, 1), 6: (2, 3),
    7: (3, 0), 8: (3, 2),
}
RC_TO_SQUARE = {v: k for k, v in SQUARE_TO_RC.items()}

ROWS = 4
COLS = 4


def owner(piece):
    if piece in (BLACK_MAN, BLACK_KING):
        return BLACK
    if piece in (WHITE_MAN, WHITE_KING):
        return WHITE
    return EMPTY


def is_king(piece):
    return piece in (BLACK_KING, WHITE_KING)


def promote(piece, sq):
    r, _ = SQUARE_TO_RC[sq]
    if piece == BLACK_MAN and r == ROWS - 1:
        return BLACK_KING
    if piece == WHITE_MAN and r == 0:
        return WHITE_KING
    return piece


def _build_adjacency():
    adj = {}
    jumps = {}
    for sq, (r, c) in SQUARE_TO_RC.items():
        adj[sq] = {}
        jumps[sq] = {}
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in RC_TO_SQUARE:
                adj[sq][(dr, dc)] = RC_TO_SQUARE[(nr, nc)]
            jr, jc = r + 2 * dr, c + 2 * dc
            if (nr, nc) in RC_TO_SQUARE and (jr, jc) in RC_TO_SQUARE:
                jumps[sq][(dr, dc)] = (RC_TO_SQUARE[(nr, nc)], RC_TO_SQUARE[(jr, jc)])
    return adj, jumps


ADJACENT, JUMP_TARGETS = _build_adjacency()


def _move_dirs(piece):
    if piece == BLACK_MAN:
        return [(1, -1), (1, 1)]
    if piece == WHITE_MAN:
        return [(-1, -1), (-1, 1)]
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class Board:
    def __init__(self, squares=None, current_player=BLACK):
        if squares is None:
            squares = self._initial()
        self.squares = list(squares)
        self.current_player = current_player

    def _initial(self):
        sq = [EMPTY] * (NUM_SQUARES + 1)
        sq[1] = BLACK_MAN
        sq[2] = BLACK_MAN
        sq[7] = WHITE_MAN
        sq[8] = WHITE_MAN
        return sq

    def copy(self):
        return Board(self.squares, self.current_player)

    def get(self, sq):
        return self.squares[sq]

    def opponent(self):
        return WHITE if self.current_player == BLACK else BLACK

    def _simple_moves(self):
        moves = []
        for sq in range(1, NUM_SQUARES + 1):
            piece = self.squares[sq]
            if owner(piece) != self.current_player:
                continue
            for d in _move_dirs(piece):
                if d in ADJACENT[sq]:
                    dst = ADJACENT[sq][d]
                    if self.squares[dst] == EMPTY:
                        moves.append((sq, [dst]))
        return moves

    def _find_jumps(self, sq, piece, path, captured, results):
        found = False
        opp = WHITE if owner(piece) == BLACK else BLACK
        for d in _move_dirs(piece):
            if d not in JUMP_TARGETS[sq]:
                continue
            mid, land = JUMP_TARGETS[sq][d]
            if owner(self.squares[mid]) == opp and mid not in captured and self.squares[land] == EMPTY:
                found = True
                promoted = promote(piece, land)
                self._find_jumps(land, promoted, path + [land], captured | {mid}, results)
        if not found and path:
            results.append(path)

    def _jump_moves(self):
        moves = []
        for sq in range(1, NUM_SQUARES + 1):
            piece = self.squares[sq]
            if owner(piece) != self.current_player:
                continue
            paths = []
            self._find_jumps(sq, piece, [], set(), paths)
            for path in paths:
                moves.append((sq, path))
        return moves

    def legal_moves(self):
        jumps = self._jump_moves()
        if jumps:
            return jumps
        return self._simple_moves()

    def make_move(self, move):
        origin, steps = move
        b = self.copy()
        piece = b.squares[origin]
        b.squares[origin] = EMPTY
        prev = origin
        for dst in steps:
            pr, pc = SQUARE_TO_RC[prev]
            dr, dc = SQUARE_TO_RC[dst]
            if abs(dr - pr) == 2:
                mid_r, mid_c = (pr + dr) // 2, (pc + dc) // 2
                mid_sq = RC_TO_SQUARE[(mid_r, mid_c)]
                b.squares[mid_sq] = EMPTY
            prev = dst
        final = steps[-1]
        piece = promote(piece, final)
        b.squares[final] = piece
        b.current_player = b.opponent()
        return b

    def is_terminal(self):
        if not any(owner(self.squares[s]) == BLACK for s in range(1, NUM_SQUARES + 1)):
            return True
        if not any(owner(self.squares[s]) == WHITE for s in range(1, NUM_SQUARES + 1)):
            return True
        return len(self.legal_moves()) == 0

    def result(self):
        has_black = any(owner(self.squares[s]) == BLACK for s in range(1, NUM_SQUARES + 1))
        has_white = any(owner(self.squares[s]) == WHITE for s in range(1, NUM_SQUARES + 1))
        if not has_black:
            return -1
        if not has_white:
            return 1
        if len(self.legal_moves()) == 0:
            return -1 if self.current_player == BLACK else 1
        return 0

    def display(self):
        chars = {EMPTY: '.', BLACK_MAN: 'b', WHITE_MAN: 'w', BLACK_KING: 'B', WHITE_KING: 'W'}
        for r in range(ROWS):
            row = []
            for c in range(COLS):
                if (r, c) in RC_TO_SQUARE:
                    row.append(chars[self.squares[RC_TO_SQUARE[(r, c)]]])
                else:
                    row.append(' ')
            print(' '.join(row))
        print(f"Turn: {'BLACK' if self.current_player == BLACK else 'WHITE'}")


def _board_key(board):
    return (tuple(board.squares), board.current_player)


_tt = {}


def minimax(board, alpha=-2, beta=2, depth=0, max_depth=60, seen=None):
    if seen is None:
        seen = set()

    key = _board_key(board)
    if key in seen:
        return 0, None
    if board.is_terminal() or depth >= max_depth:
        return board.result(), None
    if key in _tt:
        return _tt[key]

    seen = seen | {key}
    moves = board.legal_moves()
    if board.current_player == BLACK:
        best_val = -2
        best_move = None
        for move in moves:
            val, _ = minimax(board.make_move(move), alpha, beta, depth + 1, max_depth, seen)
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        _tt[key] = (best_val, best_move)
        return best_val, best_move
    else:
        best_val = 2
        best_move = None
        for move in moves:
            val, _ = minimax(board.make_move(move), alpha, beta, depth + 1, max_depth, seen)
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, val)
            if alpha >= beta:
                break
        _tt[key] = (best_val, best_move)
        return best_val, best_move
