import chess
from stockfish import Stockfish

engine = Stockfish(
    "stockfish/stockfish-windows-x86-64.exe"
)

def value(board):
    engine.set_fen_position(board.fen())

    result = engine.get_evaluation()

    if result["type"] == "cp":
        return result["value"]

    # mate scores
    return 100000 if result["value"] > 0 else -100000