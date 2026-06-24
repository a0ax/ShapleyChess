from datasets import load_from_disk

ds = load_from_disk("data/tournament-games")

print(ds.column_names)
print(ds[0])