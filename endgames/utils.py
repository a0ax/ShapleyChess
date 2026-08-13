from __future__ import annotations

from pathlib import Path

import chess


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "tournament-games"
OPENINGS_FILE = REPO_ROOT / "data" / "identified_openings.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def total_piece_count(board: chess.Board) -> int:
    return len(board.piece_map())


def non_king_piece_squares(board: chess.Board) -> list[int]:
    return [
        square
        for square, piece in board.piece_map().items()
        if piece.piece_type != chess.KING
    ]


def material_value(board: chess.Board) -> int:
    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score


def piece_descriptor(piece: chess.Piece, square: int) -> tuple[int, int, int]:
    return (int(piece.color), int(piece.piece_type), int(square))


def structure_signature(board: chess.Board) -> tuple:
    pieces = tuple(
        sorted(
            piece_descriptor(piece, square)
            for square, piece in board.piece_map().items()
        )
    )
    return (
        int(board.turn),
        board.king(chess.WHITE),
        board.king(chess.BLACK),
        pieces,
    )


def feature_vector(board: chess.Board) -> tuple[float, ...]:
    counts = {
        (chess.WHITE, chess.PAWN): 0,
        (chess.WHITE, chess.KNIGHT): 0,
        (chess.WHITE, chess.BISHOP): 0,
        (chess.WHITE, chess.ROOK): 0,
        (chess.WHITE, chess.QUEEN): 0,
        (chess.BLACK, chess.PAWN): 0,
        (chess.BLACK, chess.KNIGHT): 0,
        (chess.BLACK, chess.BISHOP): 0,
        (chess.BLACK, chess.ROOK): 0,
        (chess.BLACK, chess.QUEEN): 0,
    }

    for piece in board.piece_map().values():
        if piece.piece_type == chess.KING:
            continue
        counts[(piece.color, piece.piece_type)] += 1

    def square_value(square: int | None) -> float:
        return -1.0 if square is None else float(square) / 63.0

    return (
        float(total_piece_count(board)),
        float(material_value(board)) / 100.0,
        float(board.legal_moves.count()) / 10.0,
        1.0 if board.turn == chess.WHITE else 0.0,
        square_value(board.king(chess.WHITE)),
        square_value(board.king(chess.BLACK)),
        float(counts[(chess.WHITE, chess.PAWN)]),
        float(counts[(chess.WHITE, chess.KNIGHT)]),
        float(counts[(chess.WHITE, chess.BISHOP)]),
        float(counts[(chess.WHITE, chess.ROOK)]),
        float(counts[(chess.WHITE, chess.QUEEN)]),
        float(counts[(chess.BLACK, chess.PAWN)]),
        float(counts[(chess.BLACK, chess.KNIGHT)]),
        float(counts[(chess.BLACK, chess.BISHOP)]),
        float(counts[(chess.BLACK, chess.ROOK)]),
        float(counts[(chess.BLACK, chess.QUEEN)]),
    )


def l1_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right))


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
