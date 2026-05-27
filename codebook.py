PAD_TOKEN = '<PAD>'

INPUT_TOKENS = [PAD_TOKEN, 'X', 'O', '1', '2', '3', '/', ' ']
TOKEN_TO_ID = {t: i for i, t in enumerate(INPUT_TOKENS)}
ID_TO_TOKEN = {i: t for i, t in enumerate(INPUT_TOKENS)}
INPUT_VOCAB_SIZE = len(INPUT_TOKENS)

NUM_MOVES = 9
MAX_SEQ_LEN = 12


def encode_fen(fen):
    return [TOKEN_TO_ID[ch] for ch in fen]


def decode_ids(ids):
    return ''.join(ID_TO_TOKEN[i] for i in ids)


def pad_sequence(token_ids, max_len=MAX_SEQ_LEN):
    padded = token_ids + [TOKEN_TO_ID[PAD_TOKEN]] * (max_len - len(token_ids))
    return padded[:max_len]
