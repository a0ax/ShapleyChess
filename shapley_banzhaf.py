import random
from collections import defaultdict
import chess

def compute_shapley_banzhaf(board, eval_func, num_samples=300):
    """
    Compute approximate Shapley and Banzhaf values using only eval_func.
    eval_func must accept a chess.Board and return a numeric score.
    Raises an exception if any evaluation fails (no fallback).
    """
    # Collect non‑king pieces
    pieces = [(sq, board.piece_at(sq).color) for sq in chess.SQUARES
              if board.piece_at(sq) and board.piece_at(sq).piece_type != chess.KING]
    n = len(pieces)
    if n == 0:
        return {}, {}

    shapley_sums = defaultdict(float)
    banzhaf_sums = defaultdict(float)

    # ---------- Shapley ----------
    for _ in range(num_samples):
        perm = random.sample(pieces, n)
        current_board = board.copy()
        for sq, _ in pieces:
            current_board.remove_piece_at(sq)
        v_prev = eval_func(current_board)

        for sq, color in perm:
            piece = board.piece_at(sq)
            current_board.set_piece_at(sq, piece)
            v_curr = eval_func(current_board)
            shapley_sums[sq] += v_curr - v_prev
            v_prev = v_curr

    # ---------- Banzhaf ----------
    for _ in range(num_samples):
        subset = set()
        sub_board = board.copy()
        for sq, _ in pieces:
            sub_board.remove_piece_at(sq)
        for sq, color in pieces:
            if random.random() < 0.5:
                subset.add(sq)
                sub_board.set_piece_at(sq, board.piece_at(sq))

        v_base = eval_func(sub_board)

        for sq, color in pieces:
            if sq in subset:
                sub_board.remove_piece_at(sq)
                v_new = eval_func(sub_board)
                marginal = v_base - v_new
                sub_board.set_piece_at(sq, board.piece_at(sq))
            else:
                sub_board.set_piece_at(sq, board.piece_at(sq))
                v_new = eval_func(sub_board)
                marginal = v_new - v_base
                sub_board.remove_piece_at(sq)
            banzhaf_sums[sq] += marginal

    # Average
    shapley = {sq: val / num_samples for sq, val in shapley_sums.items()}
    banzhaf = {sq: val / num_samples for sq, val in banzhaf_sums.items()}
    return shapley, banzhaf