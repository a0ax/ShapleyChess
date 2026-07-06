import json
import chess
from eval import value
from game_utils import load_game_by_index, get_position_after_ply
from shapley_banzhaf import compute_shapley_banzhaf

OPENINGS_FILE = "data/identified_openings.json"
MIDGAME_PLY = 20
NUM_SAMPLES = 300   # adjust as needed

# Load the first Italian Game
with open(OPENINGS_FILE, "r") as f:
    identified = json.load(f)
indices = identified.get("Italian Game", [])
if not indices:
    raise RuntimeError("No Italian Game found.")

game_idx = indices[0]
pgn_game = load_game_by_index(game_idx)
board = get_position_after_ply(pgn_game, MIDGAME_PLY)

print(f"Position after {MIDGAME_PLY} plies:\n{board.fen()}\n{board}")

# Compute Shapley and Banzhaf (pure Stockfish)
try:
    shapley, banzhaf = compute_shapley_banzhaf(board, value, num_samples=NUM_SAMPLES)
except Exception as e:
    print(f"Computation failed: {e}")
    exit(1)

# Team totals
def team_sum(values, board, color):
    return sum(v for sq, v in values.items() if board.piece_at(sq).color == color)

white_shap = team_sum(shapley, board, chess.WHITE)
black_shap = team_sum(shapley, board, chess.BLACK)
white_ban = team_sum(banzhaf, board, chess.WHITE)
black_ban = team_sum(banzhaf, board, chess.BLACK)

print("\n--- Team Shapley ---")
print(f"White: {white_shap:.2f}, Black: {black_shap:.2f}, Sum: {white_shap+black_shap:.2f}")

print("\n--- Team Banzhaf ---")
print(f"White: {white_ban:.2f}, Black: {black_ban:.2f}, Sum: {white_ban+black_ban:.2f}")

# Individual Shapley (top 10)
sorted_shap = sorted(shapley.items(), key=lambda x: -abs(x[1]))
print("\n--- Individual Shapley (top 10) ---")
for sq, val in sorted_shap[:10]:
    piece = board.piece_at(sq)
    print(f"{piece} on {chess.SQUARE_NAMES[sq]}: {val:.2f}")

# Individual Banzhaf (top 10)
sorted_ban = sorted(banzhaf.items(), key=lambda x: -abs(x[1]))
print("\n--- Individual Banzhaf (top 10) ---")
for sq, val in sorted_ban[:10]:
    piece = board.piece_at(sq)
    print(f"{piece} on {chess.SQUARE_NAMES[sq]}: {val:.2f}")

# Evaluate the original position (should succeed)
try:
    eval_val = value(board)
    print(f"\nStockfish evaluation of full position: {eval_val:.2f} cp")
except Exception as e:
    print(f"\nFull position evaluation failed: {e}")