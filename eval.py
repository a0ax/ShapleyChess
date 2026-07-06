import os
import chess
from stockfish import Stockfish

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "stockfish", "stockfish-windows-x86-64-avx2.exe")

# Global engine, will be (re)initialised as needed
_engine = None

def _get_engine():
    """Return a Stockfish engine instance, creating one if needed."""
    global _engine
    if _engine is None:
        _engine = Stockfish(ENGINE_PATH, parameters={"Threads": 1, "Hash": 64})
    return _engine

def _reset_engine():
    """Close and re‑create the engine (useful after a failure)."""
    global _engine
    if _engine is not None:
        try:
            _engine.send_quit_command()
        except Exception:
            pass
    _engine = None

def value(board):
    """
    Evaluate a chess board using Stockfish (centipawns, white positive).
    Raises an exception if evaluation fails after one retry.
    No material fallback – if you want that, handle it outside.
    """
    # First attempt
    try:
        engine = _get_engine()
        engine.set_fen_position(board.fen())
        result = engine.get_evaluation()
        if result["type"] == "cp":
            return result["value"]
        # mate score: convert to cp equivalent
        return 100000 if result["value"] > 0 else -100000
    except Exception:
        # If engine failed, reset and try once more
        _reset_engine()
        try:
            engine = _get_engine()
            engine.set_fen_position(board.fen())
            result = engine.get_evaluation()
            if result["type"] == "cp":
                return result["value"]
            return 100000 if result["value"] > 0 else -100000
        except Exception as e:
            # Re‑raise the exception – no fallback
            raise RuntimeError(f"Stockfish evaluation failed for position {board.fen()}") from e