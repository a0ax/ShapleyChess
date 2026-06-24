import chess
from eval import value

positions = {
    "starting position": chess.Board(),

    "white up a queen": chess.Board(
        "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"
    ),

    "black up a queen": chess.Board(
        "q3k3/8/8/8/8/8/8/4K3 b - - 0 1"
    ),

    "checkmate": chess.Board(
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    )
}


for name, board in positions.items():

    print("\n---", name, "---")
    print(board)

    print("Evaluation:", value(board))