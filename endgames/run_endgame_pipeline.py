from __future__ import annotations

import json
import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
EXTRACTED_FILE = OUTPUT_DIR / "endgame_candidates.json"
EXACT_FILE = OUTPUT_DIR / "exact_shapley.json"


def run_script(script_name: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / script_name), *args]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def summarize_exact_results(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    openings = [name for name in results.keys() if name != "_metadata"]
    total_positions = sum(len(results[name]) for name in openings)

    print()
    print(f"Exact Shapley output: {path}")
    print(f"Openings with data: {len(openings)}")
    print(f"Total extracted positions: {total_positions}")

    for opening_name in openings[:10]:
        positions = results[opening_name]
        print(f"- {opening_name}: {len(positions)} position(s)")
        for position in positions[:2]:
            print(
                f"  ply {position['ply']}, total pieces {position['total_pieces']}, evaluation {position['evaluation']:.1f} cp"
            )
            shapley = position.get("shapley", {})
            if shapley:
                top_piece = max(shapley.items(), key=lambda item: abs(item[1]))
                print(f"  top Shapley piece: {top_piece[0]} = {top_piece[1]:.1f} cp")


def main() -> None:
    print("Running endgame extraction...")
    run_script("extract_endgames.py")

    print("Running exact Shapley computation...")
    run_script("exact_endgame_shapley.py")

    print()
    print(f"Extracted positions: {EXTRACTED_FILE}")
    print(f"Exact Shapley data: {EXACT_FILE}")

    if EXACT_FILE.exists():
        summarize_exact_results(EXACT_FILE)


if __name__ == "__main__":
    main()
