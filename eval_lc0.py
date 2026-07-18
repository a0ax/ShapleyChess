import os
import re
import subprocess
import chess

LC0_PATH = os.path.join(
    os.path.dirname(__file__),
    "leela-cuda",
    "lc0.exe",
)

WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "leela-cuda",
    "791556.pb.gz",
)

_ENGINE = None


def _read_until(predicate):

    global _ENGINE

    while True:

        line = _ENGINE.stdout.readline()

        if not line:
            raise RuntimeError(
                "lc0 terminated unexpectedly."
            )

        line = line.strip()

        if predicate(line):
            return line


def _initialize():

    global _ENGINE

    if _ENGINE is not None:
        return

    _ENGINE = subprocess.Popen(
        [
            LC0_PATH,

            "--weights=" + WEIGHTS_PATH,

            "--backend=cuda",

            "--threads=1",

            "--verbose-move-stats=false",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    def send(cmd):

        _ENGINE.stdin.write(cmd + "\n")
        _ENGINE.stdin.flush()

    send("uci")

    _read_until(
        lambda line: line == "uciok"
    )

    send("isready")

    _read_until(
        lambda line: line == "readyok"
    )


def reset_engine():

    global _ENGINE

    if _ENGINE is not None:

        try:

            _ENGINE.stdin.write("quit\n")
            _ENGINE.stdin.flush()

            _ENGINE.wait(timeout=5)

        except Exception:
            pass

    _ENGINE = None


_SCORE_RE = re.compile(
    r"score cp (-?\d+)"
)

_MATE_RE = re.compile(
    r"score mate (-?\d+)"
)


def value(
    board: chess.Board,
    nodes: int = 1,
):

    """
    Evaluate a position with LC0.

    Returns:

        centipawns from White's perspective.
    """

    global _ENGINE

    try:

        _initialize()

        fen = board.fen()

        _ENGINE.stdin.write(
            f"position fen {fen}\n"
        )

        _ENGINE.stdin.write(
            f"go nodes {nodes}\n"
        )

        _ENGINE.stdin.flush()

        bestmove_seen = False

        score = None

        while not bestmove_seen:

            line = _ENGINE.stdout.readline()

            if not line:
                raise RuntimeError(
                    "lc0 crashed."
                )

            line = line.strip()

            cp_match = _SCORE_RE.search(
                line
            )

            if cp_match:

                score = int(
                    cp_match.group(1)
                )

            mate_match = _MATE_RE.search(
                line
            )

            if mate_match:

                mate = int(
                    mate_match.group(1)
                )

                score = (
                    100000
                    if mate > 0
                    else -100000
                )

            if line.startswith(
                "bestmove"
            ):

                bestmove_seen = True

        if score is None:

            raise RuntimeError(
                "No score returned."
            )

        return score

    except Exception:

        reset_engine()

        raise