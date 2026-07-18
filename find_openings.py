import io
import json
import logging

import chess
import chess.pgn
from datasets import load_from_disk

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

MAX_GAMES = -1  # -1 = all games
DATASET_PATH = "data/tournament-games"
OUTPUT_FILE = "data/identified_openings.json"

logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

# ----------------------------------------------------------------------
# Openings (SAN)
# ----------------------------------------------------------------------

OPENINGS = [
    # Open games (1. e4 e5)
    ("Italian Game", ["e4", "e5", "Nf3", "Nc6", "Bc4"]),
    ("Giuoco Piano", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]),
    ("Evans Gambit", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "b4"]),
    ("Two Knights Defense", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"]),
    ("Ruy Lopez", ["e4", "e5", "Nf3", "Nc6", "Bb5"]),
    ("Berlin Defense", ["e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6"]),
    ("Exchange Ruy Lopez", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6"]),
    ("Scotch Game", ["e4", "e5", "Nf3", "Nc6", "d4"]),
    ("Scotch Gambit", ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Bc4"]),
    ("Vienna Game", ["e4", "e5", "Nc3"]),
    ("King's Gambit", ["e4", "e5", "f4"]),
    ("Center Game", ["e4", "e5", "d4"]),
    ("Danish Gambit", ["e4", "e5", "d4", "exd4", "c3"]),
    ("Ponziani Opening", ["e4", "e5", "Nf3", "Nc6", "c3"]),
    ("Four Knights Game", ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6"]),
    ("Petrov Defense", ["e4", "e5", "Nf3", "Nf6"]),
    ("Philidor Defense", ["e4", "e5", "Nf3", "d6"]),

    # Sicilian
    ("Sicilian Defense", ["e4", "c5"]),
    ("Open Sicilian", ["e4", "c5", "Nf3", "d6", "d4"]),
    ("Najdorf Variation", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]),
    ("Dragon Variation", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"]),
    ("Accelerated Dragon", ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "g6"]),
    ("Classical Sicilian", ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "Nf6"]),
    ("Scheveningen", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e6"]),
    ("Sveshnikov", ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e5"]),
    ("Kalashnikov", ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "e5"]),
    ("Taimanov", ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "Nc6"]),
    ("Kan Variation", ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "a6"]),
    ("Smith-Morra Gambit", ["e4", "c5", "d4", "cxd4", "c3"]),
    ("Alapin Sicilian", ["e4", "c5", "c3"]),
    ("Closed Sicilian", ["e4", "c5", "Nc3"]),
    ("Grand Prix Attack", ["e4", "c5", "Nc3", "Nc6", "f4"]),

    # French
    ("French Defense", ["e4", "e6"]),
    ("Advance French", ["e4", "e6", "d4", "d5", "e5"]),
    ("Exchange French", ["e4", "e6", "d4", "d5", "exd5"]),
    ("Tarrasch Variation", ["e4", "e6", "d4", "d5", "Nd2"]),
    ("Winawer Variation", ["e4", "e6", "d4", "d5", "Nc3", "Bb4"]),
    ("Classical French", ["e4", "e6", "d4", "d5", "Nc3", "Nf6"]),

    # Caro-Kann
    ("Caro-Kann Defense", ["e4", "c6"]),
    ("Advance Caro-Kann", ["e4", "c6", "d4", "d5", "e5"]),
    ("Exchange Caro-Kann", ["e4", "c6", "d4", "d5", "exd5"]),
    ("Classical Caro-Kann", ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4"]),
    ("Panov Attack", ["e4", "c6", "d4", "d5", "exd5", "cxd5", "c4"]),

    # Indian defenses
    ("King's Indian Defense", ["d4", "Nf6", "c4", "g6"]),
    ("Classical King's Indian", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"]),
    ("Grünfeld Defense", ["d4", "Nf6", "c4", "g6", "Nc3", "d5"]),
    ("Nimzo-Indian Defense", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"]),
    ("Queen's Indian Defense", ["d4", "Nf6", "c4", "e6", "Nf3", "b6"]),
    ("Bogo-Indian Defense", ["d4", "Nf6", "c4", "e6", "Nf3", "Bb4+"]),
    ("Benoni Defense", ["d4", "Nf6", "c4", "c5"]),
    ("Modern Benoni", ["d4", "Nf6", "c4", "c5", "d5"]),
    ("Benko Gambit", ["d4", "Nf6", "c4", "c5", "d5", "b5"]),
    ("Budapest Gambit", ["d4", "Nf6", "c4", "e5"]),

    # Queen's pawn openings
    ("Queen's Gambit", ["d4", "d5", "c4"]),
    ("Queen's Gambit Accepted", ["d4", "d5", "c4", "dxc4"]),
    ("Queen's Gambit Declined", ["d4", "d5", "c4", "e6"]),
    ("Slav Defense", ["d4", "d5", "c4", "c6"]),
    ("Semi-Slav Defense", ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6"]),
    ("London System", ["d4", "d5", "Bf4"]),
    ("Colle System", ["d4", "d5", "Nf3", "Nf6", "e3"]),
    ("Catalan Opening", ["d4", "Nf6", "c4", "e6", "g3"]),
    ("Trompowsky Attack", ["d4", "Nf6", "Bg5"]),
    ("Veresov Attack", ["d4", "Nf6", "Nc3", "d5", "Bg5"]),

    # Flank openings
    ("English Opening", ["c4"]),
    ("English, Symmetrical", ["c4", "c5"]),
    ("Réti Opening", ["Nf3"]),
    ("Bird Opening", ["f4"]),
    ("Dutch Defense", ["d4", "f5"]),
    ("Stonewall Dutch", ["d4", "f5", "g3", "Nf6", "Bg2", "e6"]),
    ("Leningrad Dutch", ["d4", "f5", "g3", "Nf6", "Bg2", "g6"]),
    ("Polish Opening", ["b4"]),
    ("Larsen Opening", ["b3"]),
    ("King's Fianchetto Opening", ["g3"]),
    ("Modern Defense", ["e4", "g6"]),
    ("Pirc Defense", ["e4", "d6", "d4", "Nf6"]),
    ("Alekhine Defense", ["e4", "Nf6"]),
    ("Scandinavian Defense", ["e4", "d5"]),
    ("Owen's Defense", ["e4", "b6"]),
]

# ----------------------------------------------------------------------
# Convert SAN openings to UCI
# ----------------------------------------------------------------------

def san_pattern_to_uci(san_moves):
    board = chess.Board()
    uci_moves = []

    for san in san_moves:
        move = board.parse_san(san)
        uci_moves.append(move.uci())
        board.push(move)

    return uci_moves


OPENINGS = [
    (name, san_pattern_to_uci(pattern))
    for name, pattern in OPENINGS
]

OPENINGS.sort(key=lambda x: len(x[1]), reverse=True)

MAX_OPENING_LENGTH = max(
    len(pattern)
    for _, pattern in OPENINGS
)

# ----------------------------------------------------------------------
# Build opening trie
# ----------------------------------------------------------------------

def build_trie(openings):
    trie = {}

    for name, moves in openings:
        node = trie

        for move in moves:
            node = node.setdefault(move, {})

        node.setdefault("_openings", []).append(name)

    return trie


OPENING_TRIE = build_trie(OPENINGS)

# ----------------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------------

print("Loading dataset...")

ds = load_from_disk(DATASET_PATH)

print(f"Loaded {len(ds):,} games.")

identified = {
    name: []
    for name, _ in OPENINGS
}

bad_games = 0

# ----------------------------------------------------------------------
# Scan games
# ----------------------------------------------------------------------

for idx, game_data in enumerate(ds):

    if MAX_GAMES != -1 and idx >= MAX_GAMES:
        break

    if idx % 10000 == 0:
        print(
            f"{idx:,} games processed "
            f"({bad_games:,} bad games)"
        )

    movetext = game_data.get("movetext", "")

    if not movetext:
        bad_games += 1
        continue

    try:
        game = chess.pgn.read_game(
            io.StringIO(movetext)
        )

        if game is None:
            bad_games += 1
            continue

        node = OPENING_TRIE

        for ply, move in enumerate(game.mainline_moves()):

            uci = move.uci()

            if uci not in node:
                break

            node = node[uci]

            if "_openings" in node:
                for opening_name in node["_openings"]:
                    identified[opening_name].append(idx)

            if ply + 1 >= MAX_OPENING_LENGTH:
                break

    except Exception:
        bad_games += 1

# ----------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------

print("Saving results...")

with open(OUTPUT_FILE, "w") as f:
    json.dump(identified, f)

print(f"\nSaved openings to {OUTPUT_FILE}\n")

for name, indices in sorted(
    identified.items(),
    key=lambda x: len(x[1]),
    reverse=True
):
    print(f"{name:<30} {len(indices):>10,}")

print(f"\nBad games: {bad_games:,}")