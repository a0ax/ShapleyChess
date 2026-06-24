import chess
from datasets import load_from_disk
import json

ds = load_from_disk("data/tournament-games")

positions = []

for game in ds:
    board = chess.Board()

    for move in game["moves"]:
        board.push_san(move)
        positions.append(board.fen())


with open("data/positions.json","w") as f:
    json.dump(positions,f)