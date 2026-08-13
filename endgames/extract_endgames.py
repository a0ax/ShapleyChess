from __future__ import annotations

import argparse
import io
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


_WORKER_DATASET = None
_WORKER_GAME_TO_OPENINGS = None


def worker_init() -> None:
    global _WORKER_DATASET, _WORKER_GAME_TO_OPENINGS
    with OPENINGS_FILE.open("r", encoding="utf-8") as handle:
        openings = json.load(handle)
    _WORKER_GAME_TO_OPENINGS = invert_openings(openings)
    _WORKER_DATASET = load_from_disk(str(DATASET_PATH))


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


def invert_openings(openings: dict[str, list[int]]) -> dict[int, list[str]]:
    game_to_openings: dict[int, list[str]] = {}
    for opening_name, game_indices in openings.items():
        for game_index in game_indices:
            game_to_openings.setdefault(game_index, []).append(opening_name)
    return game_to_openings


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


def collect_candidates(
    relevant_game_indices: list[int],
    opening_names: list[str],
    min_total_pieces: int,
    max_total_pieces: int,
) -> dict[str, list[dict]]:
    candidates_by_opening: dict[str, list[dict]] = {}
    seen_fens_by_opening: dict[str, set[str]] = {}
    candidate_target_per_opening = 6

    total_games = len(relevant_game_indices)
    chunk_size = 250
    chunks = [
        relevant_game_indices[index : index + chunk_size]
        for index in range(0, total_games, chunk_size)
    ]

    worker_count = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=worker_init,
    ) as executor:
        futures = {
            executor.submit(
                process_game_chunk,
                chunk,
                min_total_pieces,
                max_total_pieces,
            ): chunk
            for chunk in chunks
        }

        for position, future in enumerate(as_completed(futures), start=1):
            chunk_result = future.result()
            for opening_name, opening_candidates in chunk_result.items():
                seen_fens = seen_fens_by_opening.setdefault(opening_name, set())
                opening_bucket = candidates_by_opening.setdefault(opening_name, [])
                for candidate in opening_candidates:
                    if candidate["fen"] in seen_fens:
                        continue
                    seen_fens.add(candidate["fen"])
                    opening_bucket.append(candidate)

            if position == 1 or position % 20 == 0 or position == len(futures):
                processed = min(position * chunk_size, total_games)
                print(f"  scanned ~{processed:,}/{total_games:,} relevant games")

            if all(
                len(candidates_by_opening.get(opening_name, [])) >= candidate_target_per_opening
                for opening_name in opening_names
            ):
                for pending_future in futures:
                    pending_future.cancel()
                print(
                    f"  reached {candidate_target_per_opening} raw candidate(s) for every opening; stopping early"
                )
                break

    return candidates_by_opening


def process_game_chunk(
    game_indices: list[int],
    min_total_pieces: int,
    max_total_pieces: int,
) -> dict[str, list[dict]]:
    assert _WORKER_DATASET is not None
    assert _WORKER_GAME_TO_OPENINGS is not None

    candidates_by_opening: dict[str, list[dict]] = {}
    seen_fens_by_opening: dict[str, set[str]] = {}

    for game_index in game_indices:
        openings_for_game = _WORKER_GAME_TO_OPENINGS.get(game_index, [])
        if not openings_for_game:
            continue

        game_data = _WORKER_DATASET[game_index]
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
            for opening_name in openings_for_game:
                seen_fens = seen_fens_by_opening.setdefault(opening_name, set())
                if fen in seen_fens:
                    continue

                seen_fens.add(fen)
                candidates_by_opening.setdefault(opening_name, []).append(
                    candidate_from_board(
                        opening_name,
                        game_index,
                        ply + 1,
                        board,
                    )
                )

    return candidates_by_opening


def main() -> None:
    args = parse_args()
    openings = load_openings()
    game_to_openings = invert_openings(openings)

    results = {
        "_metadata": {
            "min_total_pieces": args.min_total_pieces,
            "max_total_pieces": args.max_total_pieces,
            "max_per_opening": args.max_per_opening,
            "evaluator_note": "These candidates are selected by piece-count and structural diversity only.",
        }
    }

    print(f"Processing {len(game_to_openings):,} relevant games across {len(openings):,} openings...")
    candidates_by_opening = collect_candidates(
        sorted(game_to_openings),
        list(openings.keys()),
        args.min_total_pieces,
        args.max_total_pieces,
    )

    for opening_name in openings:
        candidates = candidates_by_opening.get(opening_name, [])
        if not candidates:
            print(f"{opening_name}: no qualifying positions")
            continue

        selected = select_diverse_candidates(candidates, args.max_per_opening)
        results[opening_name] = selected
        print(f"{opening_name}: kept {len(selected)} of {len(candidates)} candidate(s)")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
