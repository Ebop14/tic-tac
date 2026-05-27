from game import EMPTY, X, O

PIECE_CHARS = {X: 'X', O: 'O'}
CHAR_TO_PIECE = {'X': X, 'O': O}


def board_to_fen(board):
    fen_parts = []
    for row in range(3):
        row_str = ''
        empty_count = 0
        for col in range(3):
            sq = board.squares[row * 3 + col]
            if sq == EMPTY:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += PIECE_CHARS[sq]
        if empty_count > 0:
            row_str += str(empty_count)
        fen_parts.append(row_str)
    player_char = PIECE_CHARS[board.current_player]
    return '/'.join(fen_parts) + ' ' + player_char


def fen_to_board(fen):
    from game import Board
    parts = fen.split(' ')
    board_str, player_char = parts[0], parts[1]
    squares = []
    for ch in board_str:
        if ch == '/':
            continue
        elif ch.isdigit():
            squares.extend([EMPTY] * int(ch))
        else:
            squares.append(CHAR_TO_PIECE[ch])
    player = CHAR_TO_PIECE[player_char]
    return Board(squares, player)
