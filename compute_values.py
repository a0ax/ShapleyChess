import chess
import chess.pgn
import io
import json
import random
from collections import defaultdict
from datasets import load_from_disk

# Choose your evaluation function
from eval import value   # Stockfish
# from eval import material as value   # uncomment to test with material

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_PATH = "data/tournament-games"
OPENINGS_FILE = "data/identified_openings.json"
MIDGAME_PLY = 20
NUM_SAMPLES = 300

# ----------------------------------------------------------------------
# Load the first Italian Game
# ----------------------------------------------------------------------
with open(OPENINGS_FILE, "r") as f:
    identified = json.load(f)

italian_indices = identified.get("Italian Game", [])
if not italian_indices:
    raise RuntimeError("No Italian Game found.")

game_index = italian_indices[0]

ds = load_from_disk(DATASET_PATH)
game_data = ds[game_index]
pgn_game = chess.pgn.read_game(io.StringIO(game_data["movetext"]))
board = pgn_game.board()

ply = 0
for move in pgn_game.mainline_moves():
    if ply >= MIDGAME_PLY:
        break
    board.push(move)
    ply += 1

print(f"Position after {ply} plies:\n{board.fen()}\n{board}")

# ----------------------------------------------------------------------
# Evaluation wrapper (with fallback)
# ----------------------------------------------------------------------
def evaluate(b):
    try:
        return value(b)
    except Exception:
        # Instead of 0, return a large penalty to give signal
        # For example, if black is in check, it's bad for black, etc.
        # But a simple heuristic: if the position is illegal, return -10000 or +10000
        # Let's try to detect check:
        if b.is_check():
            # Return a big advantage to the side not in check
            # But we don't know which side, we can evaluate a material count
            # For simplicity, fallback to material
            return material(b)
        return material(b)

# Simple material evaluation (fallback)
def material(b):
    vals = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    score = 0
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p:
            score += vals[p.piece_type] if p.color == chess.WHITE else -vals[p.piece_type]
    return score

# ----------------------------------------------------------------------
# Monte Carlo Shapley & Banzhaf
# ----------------------------------------------------------------------
def compute_values(board, num_samples=NUM_SAMPLES):
    pieces = [(sq, board.piece_at(sq).color) for sq in chess.SQUARES
              if board.piece_at(sq) and board.piece_at(sq).piece_type != chess.KING]
    n = len(pieces)
    if n == 0:
        return {}, {}

    shapley_sums = defaultdict(float)
    banzhaf_sums = defaultdict(float)

    # Shapley
    for _ in range(num_samples):
        perm = random.sample(pieces, n)
        current_board = board.copy()
        for sq, _ in pieces:
            current_board.remove_piece_at(sq)
        v_prev = evaluate(current_board)
        for sq, color in perm:
            piece = board.piece_at(sq)
            current_board.set_piece_at(sq, piece)
            v_curr = evaluate(current_board)
            shapley_sums[sq] += v_curr - v_prev
            v_prev = v_curr

    # Banzhaf
    for _ in range(num_samples):
        subset = set()
        sub_board = board.copy()
        for sq, _ in pieces:
            sub_board.remove_piece_at(sq)
        for sq, color in pieces:
            if random.random() < 0.5:
                subset.add(sq)
                sub_board.set_piece_at(sq, board.piece_at(sq))

        v_base = evaluate(sub_board)

        for sq, color in pieces:
            if sq in subset:
                sub_board.remove_piece_at(sq)
                v_new = evaluate(sub_board)
                marginal = v_base - v_new
                sub_board.set_piece_at(sq, board.piece_at(sq))
            else:
                sub_board.set_piece_at(sq, board.piece_at(sq))
                v_new = evaluate(sub_board)
                marginal = v_new - v_base
                sub_board.remove_piece_at(sq)
            banzhaf_sums[sq] += marginal

    shapley = {sq: val / num_samples for sq, val in shapley_sums.items()}
    banzhaf = {sq: val / num_samples for sq, val in banzhaf_sums.items()}
    return shapley, banzhaf

# ----------------------------------------------------------------------
# Compute and display
# ----------------------------------------------------------------------
shapley, banzhaf = compute_values(board)

white_shapley = sum(v for sq, v in shapley.items() if board.piece_at(sq).color == chess.WHITE)
black_shapley = sum(v for sq, v in shapley.items() if board.piece_at(sq).color == chess.BLACK)

white_banzhaf = sum(v for sq, v in banzhaf.items() if board.piece_at(sq).color == chess.WHITE)
black_banzhaf = sum(v for sq, v in banzhaf.items() if board.piece_at(sq).color == chess.BLACK)

print("\n--- Team Shapley ---")
print(f"White: {white_shapley:.2f}, Black: {black_shapley:.2f}, Sum: {white_shapley+black_shapley:.2f}")

print("\n--- Team Banzhaf ---")
print(f"White: {white_banzhaf:.2f}, Black: {black_banzhaf:.2f}, Sum: {white_banzhaf+black_banzhaf:.2f}")

print("\n--- Individual Shapley (top 10) ---")
sorted_shapley = sorted(shapley.items(), key=lambda x: -abs(x[1]))
for sq, val in sorted_shapley[:10]:
    piece = board.piece_at(sq)
    print(f"{piece} on {chess.SQUARE_NAMES[sq]}: {val:.2f}")

print(f"\nStockfish evaluation: {evaluate(board):.2f} cp")