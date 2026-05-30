EMPTY = 0
BLACK_MAN = 1
WHITE_MAN = 2
BLACK_KING = 3
WHITE_KING = 4

BLACK = 1
WHITE = 2

ROWS = 8
COLS = 8

# Standard 8x8 checkers. Dark (playable) squares are those where (r + c) is
# odd, numbered 1..32 left-to-right, top-to-bottom:
#   .  1  .  2  .  3  .  4
#   5  .  6  .  7  .  8  .
#   .  9  . 10  . 11  . 12
#  13  . 14  . 15  . 16  .
#   . 17  . 18  . 19  . 20
#  21  . 22  . 23  . 24  .
#   . 25  . 26  . 27  . 28
#  29  . 30  . 31  . 32  .
SQUARE_TO_RC = {}
_n = 1
for _r in range(ROWS):
    for _c in range(COLS):
        if (_r + _c) % 2 == 1:
            SQUARE_TO_RC[_n] = (_r, _c)
            _n += 1
RC_TO_SQUARE = {v: k for k, v in SQUARE_TO_RC.items()}
NUM_SQUARES = len(SQUARE_TO_RC)  # 32


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
        for s in range(1, 13):
            sq[s] = BLACK_MAN
        for s in range(21, 33):
            sq[s] = WHITE_MAN
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


# --- Heuristic, depth-limited minimax ---------------------------------------
#
# 8x8 checkers cannot be solved exhaustively (unlike the 4x4 variant), so the
# "teacher" here is alpha-beta to a fixed depth with a material + advancement
# evaluation at the leaves. Scores are from BLACK's perspective: positive is
# good for BLACK, negative good for WHITE.

WIN = 100000

_MAN = 100
_KING = 175
_ADVANCE = 6  # per-row bonus pushing men toward promotion


def evaluate(board):
    score = 0
    for sq in range(1, NUM_SQUARES + 1):
        piece = board.squares[sq]
        if piece == EMPTY:
            continue
        r = SQUARE_TO_RC[sq][0]
        if piece == BLACK_MAN:
            score += _MAN + _ADVANCE * r
        elif piece == WHITE_MAN:
            score -= _MAN + _ADVANCE * (ROWS - 1 - r)
        elif piece == BLACK_KING:
            score += _KING
        elif piece == WHITE_KING:
            score -= _KING
    return score


def _ordered_moves(board):
    moves = board.legal_moves()
    # Longer capture sequences first — improves alpha-beta pruning.
    moves.sort(key=lambda m: len(m[1]), reverse=True)
    return moves


def search(board, depth, alpha=-WIN - 1, beta=WIN + 1):
    """Return (value, best_move) for `board` looking `depth` plies ahead.

    Value is from BLACK's perspective. Terminal wins are scored near +/-WIN and
    nudged by depth so quicker wins (and slower losses) are preferred.
    """
    if board.is_terminal():
        r = board.result()
        return r * (WIN + depth), None
    if depth == 0:
        return evaluate(board), None

    moves = _ordered_moves(board)
    if board.current_player == BLACK:
        best_val = -WIN - 1
        best_move = None
        for move in moves:
            val, _ = search(board.make_move(move), depth - 1, alpha, beta)
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
        return best_val, best_move
    else:
        best_val = WIN + 1
        best_move = None
        for move in moves:
            val, _ = search(board.make_move(move), depth - 1, alpha, beta)
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, best_val)
            if alpha >= beta:
                break
        return best_val, best_move
