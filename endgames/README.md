# Endgame Shapley Pipeline

This folder contains the endgame preprocessing and exact Shapley tooling.

## What it does

- Extracts late-game positions for every opening in `data/identified_openings.json`.
- Keeps positions with 5-6 total pieces on the board, including kings.
- Selects a small, structurally diverse subset per opening.
- Computes exact Shapley values for the selected positions.

## Scripts

- `extract_endgames.py` writes `endgame_shapley/output/endgame_candidates.json`.
- `exact_endgame_shapley.py` reads that file and writes `endgame_shapley/output/exact_shapley.json`.

## Evaluation scale

The exact Shapley runner defaults to `material`, which keeps the values near the familiar centipawn scale from classical piece values.
If you want engine-based exact Shapley instead, pass `--evaluation stockfish` or `--evaluation lc0`.

## Examples

```bash
python endgame_shapley/extract_endgames.py
python endgame_shapley/exact_endgame_shapley.py
```
