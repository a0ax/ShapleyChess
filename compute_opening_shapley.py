import json
from collections import defaultdict

import chess
import numpy as np

from game_utils import load_game_by_index, get_position_after_ply
from shapley_banzhaf import compute_shapley_banzhaf
from eval_lc0 import value

# ============================================================
# Configuration
# ============================================================

OPENINGS_FILE = "data/identified_openings.json"
OUTPUT_FILE = "data/opening_analysis.json"

MIDGAME_PLY = 20

NUM_GAMES_PER_OPENING = 5
NUM_SAMPLES = 150


# ============================================================
# Utilities
# ============================================================

def entropy(values):

    vals = np.array(
        [abs(v) for v in values.values()],
        dtype=float,
    )

    total = vals.sum()

    if total == 0:
        return 0.0

    vals /= total

    return float(
        -np.sum(
            vals * np.log(vals + 1e-12)
        )
    )


def concentration(values, k=5):

    vals = np.array(
        [abs(v) for v in values.values()],
        dtype=float,
    )

    total = vals.sum()

    if total == 0:
        return 0.0

    vals = np.sort(vals)[::-1]

    return float(
        vals[:k].sum() / total
    )


def piece_name(piece_type):

    return {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }[piece_type]


def piece_key(piece):

    color = (
        "white"
        if piece.color
        else "black"
    )

    return (
        color
        + "_"
        + piece_name(
            piece.piece_type
        )
    )


# ============================================================
# Load openings
# ============================================================

with open(OPENINGS_FILE) as f:

    openings = json.load(f)


results = {}

results["_metadata"] = {
    "midgame_ply": MIDGAME_PLY,
    "num_games_per_opening": NUM_GAMES_PER_OPENING,
    "num_samples": NUM_SAMPLES,
}

# ============================================================
# Main loop
# ============================================================

for opening_name, indices in openings.items():

    if len(indices) == 0:
        continue

    print()
    print("=" * 80)
    print(opening_name)
    print("=" * 80)

    # ----------------------------------------
    # Aggregate statistics
    # ----------------------------------------

    square_totals_shapley = np.zeros(64)
    square_totals_banzhaf = np.zeros(64)

    piece_totals_shapley = defaultdict(float)
    piece_totals_banzhaf = defaultdict(float)

    evaluations = []

    mobilities = []

    shapley_entropies = []
    banzhaf_entropies = []

    shapley_concentrations = []
    banzhaf_concentrations = []

    example_fens = []

    game_results = []

    analyzed = 0

    # ----------------------------------------
    # Analyze games
    # ----------------------------------------

    for game_index in indices[:NUM_GAMES_PER_OPENING]:

        try:

            game = load_game_by_index(
                game_index
            )

            board = get_position_after_ply(
                game,
                MIDGAME_PLY,
            )

            shapley, banzhaf = (
                compute_shapley_banzhaf(
                    board,
                    value,
                    num_samples=NUM_SAMPLES,
                )
            )

            eval_cp = value(board)

            mobility = board.legal_moves.count()

            shap_entropy = entropy(
                shapley
            )

            ban_entropy = entropy(
                banzhaf
            )

            shap_conc = concentration(
                shapley
            )

            ban_conc = concentration(
                banzhaf
            )

            evaluations.append(
                eval_cp
            )

            mobilities.append(
                mobility
            )

            shapley_entropies.append(
                shap_entropy
            )

            banzhaf_entropies.append(
                ban_entropy
            )

            shapley_concentrations.append(
                shap_conc
            )

            banzhaf_concentrations.append(
                ban_conc
            )

            if len(example_fens) < 5:

                example_fens.append(
                    board.fen()
                )

            # --------------------------------
            # Aggregate square + piece stats
            # --------------------------------

            for sq, val in shapley.items():

                square_totals_shapley[
                    sq
                ] += abs(val)

                piece = board.piece_at(
                    sq
                )

                if piece:

                    piece_totals_shapley[
                        piece_key(piece)
                    ] += abs(val)

            for sq, val in banzhaf.items():

                square_totals_banzhaf[
                    sq
                ] += abs(val)

                piece = board.piece_at(
                    sq
                )

                if piece:

                    piece_totals_banzhaf[
                        piece_key(piece)
                    ] += abs(val)

            # --------------------------------
            # Save raw game data
            # --------------------------------

            game_results.append({

                "game_index":
                    game_index,

                "fen":
                    board.fen(),

                "evaluation_cp":
                    float(eval_cp),

                "mobility":
                    mobility,

                "shapley_entropy":
                    float(
                        shap_entropy
                    ),

                "banzhaf_entropy":
                    float(
                        ban_entropy
                    ),

                "top5_shapley":
                    float(
                        shap_conc
                    ),

                "top5_banzhaf":
                    float(
                        ban_conc
                    ),

                "num_pieces":
                    len(shapley),

                "shapley":
                    {
                        chess.square_name(
                            sq
                        ): float(v)
                        for sq, v
                        in shapley.items()
                    },

                "banzhaf":
                    {
                        chess.square_name(
                            sq
                        ): float(v)
                        for sq, v
                        in banzhaf.items()
                    },
            })

            analyzed += 1

            print(
                f"{analyzed}/{NUM_GAMES_PER_OPENING}"
            )

        except Exception as e:

            print(
                f"Skipped game {game_index}: {e}"
            )

    # ----------------------------------------
    # Skip empty openings
    # ----------------------------------------

    if analyzed == 0:
        continue

    # ----------------------------------------
    # Normalize aggregates
    # ----------------------------------------

    square_totals_shapley /= analyzed
    square_totals_banzhaf /= analyzed

    piece_totals_shapley = {

        k: float(v / analyzed)

        for k, v in
        piece_totals_shapley.items()
    }

    piece_totals_banzhaf = {

        k: float(v / analyzed)

        for k, v in
        piece_totals_banzhaf.items()
    }

    # ----------------------------------------
    # Save opening
    # ----------------------------------------
    results[opening_name] = {

        "num_games":
            analyzed,

        "mean_eval":
            float(
                np.mean(
                    evaluations
                )
            ),

        "std_eval":
            float(
                np.std(
                    evaluations
                )
            ),

        "mean_mobility":
            float(
                np.mean(
                    mobilities
                )
            ),

        "std_mobility":
            float(
                np.std(
                    mobilities
                )
            ),

        "mean_shapley_entropy":
            float(
                np.mean(
                    shapley_entropies
                )
            ),

        "std_shapley_entropy":
            float(
                np.std(
                    shapley_entropies
                )
            ),

        "mean_banzhaf_entropy":
            float(
                np.mean(
                    banzhaf_entropies
                )
            ),

        "std_banzhaf_entropy":
            float(
                np.std(
                    banzhaf_entropies
                )
            ),

        "mean_top5_shapley":
            float(
                np.mean(
                    shapley_concentrations
                )
            ),

        "mean_top5_banzhaf":
            float(
                np.mean(
                    banzhaf_concentrations
                )
            ),

        "piece_totals_shapley":
            piece_totals_shapley,

        "piece_totals_banzhaf":
            piece_totals_banzhaf,

        "square_heatmap_shapley":
            square_totals_shapley.tolist(),

        "square_heatmap_banzhaf":
            square_totals_banzhaf.tolist(),

        "example_fens":
            example_fens,

        "games":
            game_results,
    }


# ============================================================
# Save
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )

print()
print(
    f"Saved -> {OUTPUT_FILE}"
)