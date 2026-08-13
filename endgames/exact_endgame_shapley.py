from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import chess

from utils import (
    ensure_output_dir,
    non_king_piece_squares,
    total_piece_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact Shapley values for extracted endgames."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=ensure_output_dir() / "endgame_candidates.json",
        help="The extracted endgame candidate file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ensure_output_dir() / "exact_shapley.json",
        help="Where to write the exact Shapley results.",
    )
    parser.add_argument(
        "--evaluation",
        choices=["material", "stockfish", "lc0"],
        default="material",
        help="Evaluation function used for exact Shapley.",
    )
    return parser.parse_args()


def load_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def material_value(board: chess.Board) -> int:
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    score = 0
    for piece in board.piece_map().values():
        value = piece_values[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score


def get_evaluator(name: str):
    if name == "material":
        return material_value
    if name == "stockfish":
        from eval import value as stockfish_value

        return stockfish_value
    if name == "lc0":
        from eval_lc0 import value as lc0_value

        return lc0_value
    raise ValueError(name)


def build_subset_board(board: chess.Board, included_squares: set[int]) -> chess.Board:
    subset_board = board.copy(stack=False)
    for square in list(subset_board.piece_map().keys()):
        piece = subset_board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            subset_board.remove_piece_at(square)

    subset_board.castling_rights = chess.Bitboard(0)
    subset_board.ep_square = None

    for square in included_squares:
        subset_board.set_piece_at(square, board.piece_at(square))

    return subset_board


def exact_shapley(board: chess.Board, eval_func) -> dict[str, float]:
    pieces = non_king_piece_squares(board)
    n = len(pieces)
    if n == 0:
        return {}

    evaluations: dict[frozenset[int], float] = {}
    for subset_size in range(n + 1):
        for subset in itertools.combinations(pieces, subset_size):
            subset_key = frozenset(subset)
            subset_board = build_subset_board(board, set(subset))
            evaluations[subset_key] = float(eval_func(subset_board))

    factorial_n = math.factorial(n)
    shapley: dict[str, float] = {}

    for square in pieces:
        others = [candidate for candidate in pieces if candidate != square]
        contribution = 0.0

        for subset_size in range(n):
            weight = (
                math.factorial(subset_size)
                * math.factorial(n - subset_size - 1)
                / factorial_n
            )
            for subset in itertools.combinations(others, subset_size):
                subset_key = frozenset(subset)
                with_piece = frozenset(set(subset) | {square})
                contribution += weight * (
                    evaluations[with_piece] - evaluations[subset_key]
                )

        shapley[chess.square_name(square)] = contribution

    return shapley


def compute_results(input_data: dict, eval_func) -> dict:
    results = {
        "_metadata": {
            "evaluation": eval_func.__name__,
        }
    }

    for opening_name, positions in input_data.items():
        if opening_name == "_metadata":
            continue

        opening_results = []
        for position in positions:
            board = chess.Board(position["fen"])
            opening_results.append(
                {
                    "game_index": position["game_index"],
                    "ply": position["ply"],
                    "fen": position["fen"],
                    "total_pieces": total_piece_count(board),
                    "non_king_pieces": len(non_king_piece_squares(board)),
                    "evaluation": float(eval_func(board)),
                    "shapley": exact_shapley(board, eval_func),
                }
            )

        if opening_results:
            results[opening_name] = opening_results

    return results


def main() -> None:
    args = parse_args()
    input_data = load_input(args.input_file)
    eval_func = get_evaluator(args.evaluation)
    results = compute_results(input_data, eval_func)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved exact Shapley results to {args.output_file}")


if __name__ == "__main__":
    main()
