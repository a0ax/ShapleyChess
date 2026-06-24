import chess
import chess.pgn
import io
from datasets import load_from_disk

ds = load_from_disk("data/tournament-games")

with open("data/tournament_positions.jsonl", "w") as f:

    for game in ds:

        pgn = io.StringIO(game["movetext"])
        parsed = chess.pgn.read_game(pgn)

        if parsed is None:
            continue

        board = parsed.board()

        for move in parsed.mainline_moves():
            board.push(move)

            f.write(
                '{"fen":"' + board.fen() + '"}\n'
            )