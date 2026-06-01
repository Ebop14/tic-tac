# tic-tac

Small transformer game engines — tic-tac-toe and checkers — trained to imitate a
minimax teacher, served through a Flask web app with a live model-output panel.

Each game has its own self-contained package (board rules, notation, codebook,
model, training script, and a trained `model.pt`). The Flask app in `app.py`
loads the trained models and exposes a board UI plus a panel that visualizes the
model's per-move scores and the actual odds it plays each move.

## Games

| Variant | Folder | Board | Teacher | Model objective |
|---|---|---|---|---|
| Tic-tac-toe | `./` (root) | 3×3 | exhaustive minimax | value regression |
| Checkers (4×4) | `checkers/` | 4×4 | exhaustive minimax | move classifier |
| Checkers (8×8) | `checkers8/` | 8×8 | depth-limited minimax | value regression |
| Checkers (8×8, optional captures) | `checkers8free/` | 8×8 | depth-limited minimax | value regression |

## Training methodology

### The teacher is minimax, not self-play game outcomes

Every training label comes from a **minimax search**, never from simulated game
results or Monte-Carlo win rates.

- **Tic-tac-toe and 4×4 checkers** are small enough to **solve exhaustively** —
  minimax searches the full game tree (memoized).
- **8×8 checkers** cannot be solved, so the teacher is **alpha-beta to a fixed
  depth** (default 5) with a material + advancement heuristic at the leaves
  (man ≈ 100, king ≈ 175, +6 per row advanced, terminal win ≈ ±100000 nudged by
  depth so quicker wins are preferred).

### Two output framings: classifier vs. value regressor

**4×4 checkers — move classifier.** The label is the single minimax-best move
(one-hot). The transformer outputs logits over moves and trains with
**cross-entropy**. `softmax(logits)` is then a genuine "probability this is the
best move," and at play time the engine samples from it (temperature + top-k).

**Tic-tac-toe and 8×8 checkers — value regression.** The label is a *scalar
value per legal move* in `[-1, 1]` (from the side-to-move's perspective). The
transformer outputs a value per move through `tanh` and trains with **masked
MSE** over legal moves only (illegal squares carry no target and are excluded
from the loss). Because MSE drives each output toward its true magnitude, the
model learns the *margins* between moves — how much better one is than another —
not just which is best.

The value targets are:

- **Tic-tac-toe:** depth-discounted minimax values. Terminal nodes are ±1 / 0;
  internal nodes back up `γ · best_child` (γ = 0.9), so a win-in-3 outranks a
  win-in-7 and a slow loss beats a fast one. These are then **reward-shaped** for
  human-feeling play: +1 for taking an immediate win, −1 for leaving the
  opponent an immediate win, and all other (all-draws) play compressed near 0.
- **8×8 checkers:** the depth-limited heuristic score for each move, **squashed**
  into `[-1, 1]` via `tanh(score / 300)` (≈0.32 for a one-man edge, saturating to
  ±1 at a forced win).

### Position sourcing

- **Solved games** enumerate *every reachable non-terminal position* (deduped),
  one example each.
- **8×8 checkers** sources positions via **ε-greedy self-play** with the same
  depth-limited teacher (≈300 games, ε = 0.25) to collect a diverse set of
  unique positions, then labels each with fresh minimax searches. (Self-play
  only chooses *which positions* to train on — never the labels.)

### The model

All variants use the same shape: an **encoder-only transformer** (not
autoregressive). It reads the board as a tokenized FEN string, encodes all tokens
in parallel (bidirectional attention), mean-pools to one position vector, and a
single linear head emits **every move's score in one forward pass**. There is no
step-by-step move generation — a multi-jump capture is encoded by its
`(origin, final-destination)` pair, so even a long jump is a single output.

## Move selection at inference

For the value-regression models the engine doesn't softmax the values into a
policy. Instead it uses **value-tolerance selection** on the calibrated values:

1. Take the best legal value.
2. Keep every move within a **tolerance** of it (the pool).
3. Draw from the pool by a **temperature-softmax over the values** (better moves
   are likelier; lower temperature sharpens toward the best, higher flattens).

Tic-tac-toe additionally applies hard **tier gates** — a confident win is always
taken and a confident loss always blocked, regardless of difficulty — so only
the genuinely-near-equal moves are ever randomized.

Difficulty presets map to `(tolerance, temperature)`:

| Preset | Tic-tac-toe | 8×8 checkers |
|---|---|---|
| sharp | (0.0, 0.5) | (0.0, 0.5) — optimal |
| balanced | (0.10, 0.6) | (0.10, 0.6) |
| chill | (0.25, 1.5) | (0.50, 1.5) — loosest |

The web app's model-output panel shows each move's raw value plus the **true
probability the engine plays it** (the same temperature-softmax-over-pool
distribution used for selection), along with the active temperature/tolerance.

## Running

```bash
pip install -r requirements.txt
flask --app app run
```

Then open the served page (tic-tac-toe at `/`, checkers at `/checkers8`, etc.).

### Retraining

Each package has a `data_gen.py` (writes `data.json`) and a `train.py` (writes
`model.pt`). For example, for 8×8 checkers:

```bash
python -m checkers8.data_gen 5   # generate depth-5 value labels
python -m checkers8.train        # train the regressor
```
