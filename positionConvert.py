import chess
from datasets import load_from_disk

ds = load_from_disk("data/tournament-games")

with open("data/tournament_positions.jsonl", "w") as f:

    for game in ds:
        board = chess.Board()

        for move in game["moves"]:
            board.push_san(move)

            f.write(
                '{"fen": "' + board.fen() + '"}\n'
            )