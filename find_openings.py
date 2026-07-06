import chess
import chess.pgn
import io
import json
from datasets import load_from_disk

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MAX_GAMES = 10000
DATASET_PATH = "data/tournament-games"
OUTPUT_FILE = "data/identified_openings.json"

# Define openings as (name, list_of_san_moves)
OPENINGS = [
    ("Italian Game", ["e4", "e5", "Nf3", "Nc6", "Bc4"]),
    # ("Ruy Lopez", ["e4", "e5", "Nf3", "Nc6", "Bb5"]),
    # ("Sicilian Defense", ["e4", "c5"]),
]

# ----------------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------------
ds = load_from_disk(DATASET_PATH)

identified = {name: [] for name, _ in OPENINGS}
bad_games = 0

# ----------------------------------------------------------------------
# Scan games
# ----------------------------------------------------------------------
for idx, game_data in enumerate(ds):
    if idx >= MAX_GAMES:
        break

    movetext = game_data.get("movetext", "")
    if not movetext:
        bad_games += 1
        continue

    try:
        pgn_game = chess.pgn.read_game(io.StringIO(movetext))
        if pgn_game is None:
            bad_games += 1
            continue

        board = pgn_game.board()
        moves_san = []

        for move in pgn_game.mainline_moves():
            # Get SAN before pushing the move (correct order)
            san = board.san(move)
            moves_san.append(san)
            board.push(move)

        # Check each opening pattern
        for name, pattern in OPENINGS:
            if len(moves_san) >= len(pattern) and moves_san[:len(pattern)] == pattern:
                identified[name].append(idx)

    except Exception as e:
        # Catch any parsing error (including illegal moves, variant moves)
        bad_games += 1
        continue

# ----------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(identified, f, indent=2)

print(f"Identified openings saved to {OUTPUT_FILE}")
for name, indices in identified.items():
    print(f"{name}: {len(indices)} games")
print(f"Bad games (could not parse): {bad_games}")