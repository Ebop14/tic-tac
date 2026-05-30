from checkers8.game import (
    Board, EMPTY, BLACK, WHITE,
    BLACK_MAN, WHITE_MAN, BLACK_KING, WHITE_KING,
    NUM_SQUARES, owner, is_king,
)


def board_to_fen(board):
    w_pieces = []
    b_pieces = []
    for sq in range(1, NUM_SQUARES + 1):
        piece = board.squares[sq]
        if piece == EMPTY:
            continue
        prefix = 'K' if is_king(piece) else ''
        if owner(piece) == WHITE:
            w_pieces.append(f'{prefix}{sq}')
        else:
            b_pieces.append(f'{prefix}{sq}')
    turn = 'B' if board.current_player == BLACK else 'W'
    w_str = ','.join(w_pieces) if w_pieces else ''
    b_str = ','.join(b_pieces) if b_pieces else ''
    return f'{turn}:W{w_str}:B{b_str}'


def fen_to_board(fen):
    parts = fen.split(':')
    turn = BLACK if parts[0] == 'B' else WHITE
    squares = [EMPTY] * (NUM_SQUARES + 1)
    for part in parts[1:]:
        color = part[0]
        piece_str = part[1:]
        if not piece_str:
            continue
        for token in piece_str.split(','):
            if token.startswith('K'):
                sq = int(token[1:])
                squares[sq] = BLACK_KING if color == 'B' else WHITE_KING
            else:
                sq = int(token)
                squares[sq] = BLACK_MAN if color == 'B' else WHITE_MAN
    return Board(squares, turn)


def move_to_pdn(move, is_jump=False):
    origin, steps = move
    sep = 'x' if is_jump else '-'
    return sep.join(str(s) for s in [origin] + steps)
