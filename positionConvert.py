import chess
import chess.pgn
import io
import json
from datasets import load_from_disk

ds = load_from_disk("data/tournament-games")

MAX_GAMES = 10000
positions_written = 0
bad_games = 0

with open("data/tournament_positions.jsonl", "w") as f:

    for i, game in enumerate(ds):

        if i >= MAX_GAMES:
            break

        try:
            parsed = chess.pgn.read_game(
                io.StringIO(game["movetext"])
            )

            if parsed is None:
                bad_games += 1
                continue

            board = parsed.board()

            for move in parsed.mainline_moves():
                board.push(move)

                # skip opening positions
                #if board.fullmove_number >= 10:
                f.write(json.dumps({
                    "fen": board.fen(),
                    "white": game["White"],
                    "black": game["Black"],
                    "result": game["Result"]
                }) + "\n")

                positions_written += 1

        except Exception:
            bad_games += 1
            continue


print("positions:", positions_written)
print("bad games:", bad_games)