from datasets import load_dataset

# this is the smaller lichess tourn games dataset about 1.2 gigs
dataset = load_dataset(
    "Lichess/tournament-chess-games",
    split="train"
)

dataset.save_to_disk("data/tournament-games")