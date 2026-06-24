from stockfish import Stockfish
import os

engine = Stockfish(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "stockfish",
        "stockfish-windows-x86-64-avx2.exe"
    )
)


def value(board):
    engine.set_fen_position(board.fen())

    result = engine.get_evaluation()

    if result["type"] == "cp":
        return result["value"]

    # mate scores
    return 100000 if result["value"] > 0 else -100000