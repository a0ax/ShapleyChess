import chess
import chess.pgn
import io
from datasets import load_from_disk

DATASET_PATH = "data/tournament-games"

def load_game_by_index(idx):
    """Load a game from the dataset by its index."""
    ds = load_from_disk(DATASET_PATH)
    game_data = ds[idx]
    pgn = chess.pgn.read_game(io.StringIO(game_data["movetext"]))
    return pgn

def get_position_after_ply(pgn_game, ply):
    """Return the board after a given number of plies."""
    board = pgn_game.board()
    for i, move in enumerate(pgn_game.mainline_moves()):
        if i >= ply:
            break
        board.push(move)
    return board