from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

import chess
import chess.pgn
from datasets import load_from_disk

from utils import (
    DATASET_PATH,
    OPENINGS_FILE,
    ensure_output_dir,
    feature_vector,
    l1_distance,
    material_value,
    non_king_piece_squares,
    structure_signature,
    total_piece_count,
)


logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract diverse 5-6 piece endgames for each opening."
    )
    parser.add_argument(
        "--min-total-pieces",
        type=int,
        default=5,
        help="Minimum total piece count, including kings.",
    )
    parser.add_argument(
        "--max-total-pieces",
        type=int,
        default=6,
        help="Maximum total piece count, including kings.",
    )
    parser.add_argument(
        "--max-per-opening",
        type=int,
        default=2,
        help="Maximum number of diverse positions to keep per opening.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ensure_output_dir() / "endgame_candidates.json",
        help="Where to write the extracted positions.",
    )
    return parser.parse_args()


def load_openings() -> dict[str, list[int]]:
    with OPENINGS_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_from_board(opening_name: str, game_index: int, ply: int, board: chess.Board) -> dict:
    piece_map = board.piece_map()
    return {
        "opening": opening_name,
        "game_index": game_index,
        "ply": ply,
        "fen": board.fen(),
        "total_pieces": total_piece_count(board),
        "non_king_pieces": len(non_king_piece_squares(board)),
        "material_cp": material_value(board),
        "mobility": board.legal_moves.count(),
        "structure_signature": repr(structure_signature(board)),
        "feature_vector": list(feature_vector(board)),
        "piece_map": {
            chess.square_name(square): {
                "piece_type": piece.piece_type,
                "color": "white" if piece.color == chess.WHITE else "black",
            }
            for square, piece in piece_map.items()
        },
    }


def select_diverse_candidates(candidates: list[dict], limit: int) -> list[dict]:
    if len(candidates) <= limit:
        return candidates

    deduped: dict[str, dict] = {}
    for candidate in candidates:
        key = candidate["structure_signature"]
        if key not in deduped:
            deduped[key] = candidate

    candidates = list(deduped.values())
    if len(candidates) <= limit:
        return candidates

    vectors = [tuple(candidate["feature_vector"]) for candidate in candidates]

    selected_indices: list[int] = [
        max(
            range(len(candidates)),
            key=lambda index: (
                abs(candidates[index]["material_cp"]),
                candidates[index]["mobility"],
                -candidates[index]["ply"],
            ),
        )
    ]

    while len(selected_indices) < limit and len(selected_indices) < len(candidates):
        def score(index: int) -> tuple[float, int, int]:
            if index in selected_indices:
                return (-1.0, -1, -1)
            vector = vectors[index]
            min_distance = min(
                l1_distance(vector, vectors[selected_index])
                for selected_index in selected_indices
            )
            return (
                min_distance,
                candidates[index]["mobility"],
                -candidates[index]["ply"],
            )

        selected_indices.append(max(range(len(candidates)), key=score))

    return [candidates[index] for index in selected_indices]


def extract_for_opening(
    opening_name: str,
    game_indices: list[int],
    dataset,
    min_total_pieces: int,
    max_total_pieces: int,
    max_per_opening: int,
) -> list[dict]:
    seen_fens: set[str] = set()
    candidates: list[dict] = []

    for game_index in game_indices:
        game_data = dataset[game_index]
        movetext = game_data.get("movetext", "")
        if not movetext:
            continue

        try:
            game = chess.pgn.read_game(io.StringIO(movetext))
            if game is None:
                continue
        except Exception:
            continue

        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            board.push(move)
            total_pieces = total_piece_count(board)
            if total_pieces < min_total_pieces or total_pieces > max_total_pieces:
                continue

            fen = board.fen()
            if fen in seen_fens:
                continue

            seen_fens.add(fen)
            candidates.append(
                candidate_from_board(
                    opening_name,
                    game_index,
                    ply + 1,
                    board,
                )
            )

    return select_diverse_candidates(candidates, max_per_opening)


def main() -> None:
    args = parse_args()
    openings = load_openings()
    dataset = load_from_disk(str(DATASET_PATH))

    results = {
        "_metadata": {
            "min_total_pieces": args.min_total_pieces,
            "max_total_pieces": args.max_total_pieces,
            "max_per_opening": args.max_per_opening,
            "evaluator_note": "These candidates are selected by piece-count and structural diversity only.",
        }
    }

    for opening_name, game_indices in openings.items():
        if not game_indices:
            continue

        print(f"Extracting {opening_name}...")
        selected = extract_for_opening(
            opening_name,
            game_indices,
            dataset,
            args.min_total_pieces,
            args.max_total_pieces,
            args.max_per_opening,
        )
        if selected:
            results[opening_name] = selected
            print(f"  kept {len(selected)} position(s)")
        else:
            print("  no qualifying positions")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
