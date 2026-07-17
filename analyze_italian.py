import json
import chess
import time
from eval import value
from game_utils import load_game_by_index, get_position_after_ply
from shapley_banzhaf import compute_shapley_banzhaf

OPENINGS_FILE = "data/identified_openings.json"
MIDGAME_PLY = 20
NUM_SAMPLES = 300   # adjust as needed

# ----------------------------------------------------------------------
# Load the first Italian Game
# ----------------------------------------------------------------------
print("=" * 60)
print("ANALYZING ITALIAN GAME POSITION")
print("=" * 60)

print(f"\n[1/5] Loading openings file: {OPENINGS_FILE}")
with open(OPENINGS_FILE, "r") as f:
    identified = json.load(f)

indices = identified.get("Italian Game", [])
if not indices:
    raise RuntimeError("No Italian Game found.")

print(f"      Found {len(indices)} Italian Games in dataset")
print(f"      Using game index: {indices[0]}")

print(f"\n[2/5] Loading game from dataset...")
pgn_game = load_game_by_index(indices[0])
board = get_position_after_ply(pgn_game, MIDGAME_PLY)

print(f"      Position after {MIDGAME_PLY} plies:")
print(f"      FEN: {board.fen()}")
print(f"\n      Board:")
print(board)

# Count pieces
piece_count = {chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0, 
               chess.ROOK: 0, chess.QUEEN: 0, chess.KING: 0}
for sq in chess.SQUARES:
    piece = board.piece_at(sq)
    if piece:
        piece_count[piece.piece_type] += 1

print(f"\n      Piece count: {piece_count}")

# ----------------------------------------------------------------------
# Compute Shapley and Banzhaf
# ----------------------------------------------------------------------
print(f"\n[3/5] Computing Shapley and Banzhaf values...")
print(f"      Using {NUM_SAMPLES} Monte Carlo samples per method")
print(f"      This may take several minutes...")
start_time = time.time()

try:
    shapley, banzhaf = compute_shapley_banzhaf(board, value, num_samples=NUM_SAMPLES)
    elapsed = time.time() - start_time
    print(f"      ✓ Computation completed in {elapsed:.2f} seconds")
    print(f"      ✓ Shapley values for {len(shapley)} pieces")
    print(f"      ✓ Banzhaf values for {len(banzhaf)} pieces")
except Exception as e:
    print(f"      ✗ Computation failed after {time.time() - start_time:.2f} seconds")
    print(f"      Error: {e}")
    exit(1)

# ----------------------------------------------------------------------
# Team totals
# ----------------------------------------------------------------------
print("\n[4/5] Calculating team contributions...")

def team_sum(values, board, color):
    total = 0
    count = 0
    for sq, v in values.items():
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            total += v
            count += 1
    return total, count

white_shap, white_shap_count = team_sum(shapley, board, chess.WHITE)
black_shap, black_shap_count = team_sum(shapley, board, chess.BLACK)
white_ban, white_ban_count = team_sum(banzhaf, board, chess.WHITE)
black_ban, black_ban_count = team_sum(banzhaf, board, chess.BLACK)

print(f"\n      Shapley: White ({white_shap_count} pieces) = {white_shap:.2f}, Black ({black_shap_count} pieces) = {black_shap:.2f}")
print(f"               Sum = {white_shap + black_shap:.2f}")
print(f"      Banzhaf: White ({white_ban_count} pieces) = {white_ban:.2f}, Black ({black_ban_count} pieces) = {black_ban:.2f}")
print(f"               Sum = {white_ban + black_ban:.2f}")

# ----------------------------------------------------------------------
# Individual values
# ----------------------------------------------------------------------
print("\n[5/5] Top 10 individual contributions:")

# Individual Shapley (top 10)
sorted_shap = sorted(shapley.items(), key=lambda x: -abs(x[1]))
print("\n      --- Individual Shapley (top 10) ---")
for i, (sq, val) in enumerate(sorted_shap[:10], 1):
    piece = board.piece_at(sq)
    color = "White" if piece.color == chess.WHITE else "Black"
    piece_name = chess.piece_name(piece.piece_type).capitalize()
    print(f"      {i:2d}. {color} {piece_name} on {chess.SQUARE_NAMES[sq]}: {val:8.2f}")

# Individual Banzhaf (top 10)
sorted_ban = sorted(banzhaf.items(), key=lambda x: -abs(x[1]))
print("\n      --- Individual Banzhaf (top 10) ---")
for i, (sq, val) in enumerate(sorted_ban[:10], 1):
    piece = board.piece_at(sq)
    color = "White" if piece.color == chess.WHITE else "Black"
    piece_name = chess.piece_name(piece.piece_type).capitalize()
    print(f"      {i:2d}. {color} {piece_name} on {chess.SQUARE_NAMES[sq]}: {val:8.2f}")

# ----------------------------------------------------------------------
# Evaluate original position
# ----------------------------------------------------------------------
print("\n--- Evaluating original position with Stockfish ---")
try:
    eval_val = value(board)
    print(f"      Stockfish evaluation: {eval_val:.2f} cp")
    if eval_val > 0:
        print(f"      White is {'winning' if eval_val > 150 else 'slightly better'}")
    elif eval_val < 0:
        print(f"      Black is {'winning' if eval_val < -150 else 'slightly better'}")
    else:
        print("      Position is balanced")
except Exception as e:
    print(f"      ✗ Evaluation failed: {e}")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Game index:        {indices[0]}")
print(f"Position:          {MIDGAME_PLY} plies")
print(f"FEN:               {board.fen()}")
print(f"Samples:           {NUM_SAMPLES}")
print(f"Pieces analysed:   {len(shapley)}")
print(f"White Shapley:     {white_shap:8.2f}")
print(f"Black Shapley:     {black_shap:8.2f}")
print(f"Shapley sum:       {white_shap + black_shap:8.2f}")
print(f"White Banzhaf:     {white_ban:8.2f}")
print(f"Black Banzhaf:     {black_ban:8.2f}")
print(f"Banzhaf sum:       {white_ban + black_ban:8.2f}")
print(f"Stockfish eval:    {eval_val if eval_val else 'N/A':>8}")
print("=" * 60)