PAD_TOKEN = '<PAD>'

INPUT_TOKENS = [PAD_TOKEN, 'B', 'W', 'K', ':', ',', '1', '2', '3', '4', '5', '6', '7', '8']
TOKEN_TO_ID = {t: i for i, t in enumerate(INPUT_TOKENS)}
ID_TO_TOKEN = {i: t for i, t in enumerate(INPUT_TOKENS)}
INPUT_VOCAB_SIZE = len(INPUT_TOKENS)

NUM_SQUARES = 8
NUM_MOVES = NUM_SQUARES * NUM_SQUARES

MAX_SEQ_LEN = 24


def encode_move(origin, final_dest):
    return (origin - 1) * NUM_SQUARES + (final_dest - 1)


def decode_move(idx):
    origin = idx // NUM_SQUARES + 1
    dest = idx % NUM_SQUARES + 1
    return origin, dest


def encode_fen(fen):
    return [TOKEN_TO_ID[ch] for ch in fen]


def pad_sequence(token_ids, max_len=MAX_SEQ_LEN):
    padded = token_ids + [TOKEN_TO_ID[PAD_TOKEN]] * (max_len - len(token_ids))
    return padded[:max_len]
