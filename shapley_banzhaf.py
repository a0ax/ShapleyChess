import random
from collections import defaultdict
import chess

def remove_all_pieces_except_kings(board):
    """Remove all non-king pieces and update castling rights."""
    new_board = board.copy()
    
    # Clear castling rights first (will selectively add back)
    new_board.castling_rights = chess.Bitboard(0)
    
    # Remove all non-king pieces
    for sq in chess.SQUARES:
        piece = new_board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            new_board.remove_piece_at(sq)
    
    # For each king, check if it's in its starting position and rooks are present
    # White kingside
    if board.piece_at(chess.H1) and board.piece_at(chess.H1).piece_type == chess.ROOK and board.piece_at(chess.H1).color == chess.WHITE:
        if board.piece_at(chess.E1) and board.piece_at(chess.E1).piece_type == chess.KING and board.piece_at(chess.E1).color == chess.WHITE:
            new_board.castling_rights |= chess.BB_H1  # white kingside
    # White queenside
    if board.piece_at(chess.A1) and board.piece_at(chess.A1).piece_type == chess.ROOK and board.piece_at(chess.A1).color == chess.WHITE:
        if board.piece_at(chess.E1) and board.piece_at(chess.E1).piece_type == chess.KING and board.piece_at(chess.E1).color == chess.WHITE:
            new_board.castling_rights |= chess.BB_A1  # white queenside
    # Black kingside
    if board.piece_at(chess.H8) and board.piece_at(chess.H8).piece_type == chess.ROOK and board.piece_at(chess.H8).color == chess.BLACK:
        if board.piece_at(chess.E8) and board.piece_at(chess.E8).piece_type == chess.KING and board.piece_at(chess.E8).color == chess.BLACK:
            new_board.castling_rights |= chess.BB_H8  # black kingside
    # Black queenside
    if board.piece_at(chess.A8) and board.piece_at(chess.A8).piece_type == chess.ROOK and board.piece_at(chess.A8).color == chess.BLACK:
        if board.piece_at(chess.E8) and board.piece_at(chess.E8).piece_type == chess.KING and board.piece_at(chess.E8).color == chess.BLACK:
            new_board.castling_rights |= chess.BB_A8  # black queenside
    
    return new_board

def evaluate_with_castling(board, eval_func):
    """Evaluate a board, ensuring castling rights are consistent with piece positions."""
    try:
        return eval_func(board)
    except Exception as e:
        # If evaluation fails, the board might have inconsistent castling rights
        # Try to fix it
        fixed_board = board.copy()
        # Remove castling rights for rooks that don't exist
        # White kingside
        if not fixed_board.piece_at(chess.H1) or fixed_board.piece_at(chess.H1).piece_type != chess.ROOK:
            fixed_board.castling_rights &= ~chess.BB_H1
        # White queenside
        if not fixed_board.piece_at(chess.A1) or fixed_board.piece_at(chess.A1).piece_type != chess.ROOK:
            fixed_board.castling_rights &= ~chess.BB_A1
        # Black kingside
        if not fixed_board.piece_at(chess.H8) or fixed_board.piece_at(chess.H8).piece_type != chess.ROOK:
            fixed_board.castling_rights &= ~chess.BB_H8
        # Black queenside
        if not fixed_board.piece_at(chess.A8) or fixed_board.piece_at(chess.A8).piece_type != chess.ROOK:
            fixed_board.castling_rights &= ~chess.BB_A8
        return eval_func(fixed_board)

def compute_shapley_banzhaf(board, eval_func, num_samples=300, max_attempts=10000):
    """
    Compute approximate Shapley and Banzhaf values using only eval_func.
    Skips permutations/subsets that cause eval_func to raise an exception.
    """
    pieces = [(sq, board.piece_at(sq).color) for sq in chess.SQUARES
              if board.piece_at(sq) and board.piece_at(sq).piece_type != chess.KING]
    n = len(pieces)
    if n == 0:
        return {}, {}

    shapley_sums = defaultdict(float)
    banzhaf_sums = defaultdict(float)

    # ---------- Shapley ----------
    samples_done = 0
    attempts = 0
    while samples_done < num_samples and attempts < max_attempts:
        attempts += 1
        perm = random.sample(pieces, n)
        current_board = board.copy()
        
        # Remove all non-king pieces with proper castling rights
        for sq, _ in pieces:
            current_board.remove_piece_at(sq)
        # Fix castling rights
        current_board.castling_rights = chess.Bitboard(0)
        
        try:
            v_prev = eval_func(current_board)
        except Exception:
            continue

        valid = True
        marginal_list = []
        for sq, color in perm:
            piece = board.piece_at(sq)
            current_board.set_piece_at(sq, piece)
            
            # If it's a rook, update castling rights
            if piece.piece_type == chess.ROOK:
                # The board's castling_rights will be updated automatically when we set the piece
                # But we need to ensure consistency
                pass
                
            try:
                v_curr = eval_func(current_board)
            except Exception:
                valid = False
                break
            marginal = v_curr - v_prev
            marginal_list.append((sq, marginal))
            v_prev = v_curr

        if not valid:
            continue

        for sq, marginal in marginal_list:
            shapley_sums[sq] += marginal
        samples_done += 1

    if samples_done < num_samples:
        print(f"Warning: only {samples_done} Shapley samples out of {num_samples}")

    # ---------- Banzhaf ----------
    samples_done = 0
    attempts = 0
    while samples_done < num_samples and attempts < max_attempts:
        attempts += 1
        subset = set()
        sub_board = board.copy()
        
        # Remove all non-king pieces with proper castling rights
        for sq, _ in pieces:
            sub_board.remove_piece_at(sq)
        sub_board.castling_rights = chess.Bitboard(0)

        # Randomly include pieces
        for sq, color in pieces:
            if random.random() < 0.5:
                subset.add(sq)
                sub_board.set_piece_at(sq, board.piece_at(sq))

        # Evaluate base subset
        try:
            v_base = eval_func(sub_board)
        except Exception:
            continue

        # Compute marginals for each piece in this subset
        marginals = []
        valid = True
        for sq, color in pieces:
            if sq in subset:
                sub_board.remove_piece_at(sq)
                try:
                    v_new = eval_func(sub_board)
                except Exception:
                    valid = False
                    break
                marginal = v_base - v_new
                sub_board.set_piece_at(sq, board.piece_at(sq))
            else:
                sub_board.set_piece_at(sq, board.piece_at(sq))
                try:
                    v_new = eval_func(sub_board)
                except Exception:
                    valid = False
                    break
                marginal = v_new - v_base
                sub_board.remove_piece_at(sq)
            marginals.append((sq, marginal))

        if not valid:
            continue

        for sq, marginal in marginals:
            banzhaf_sums[sq] += marginal
        samples_done += 1

    if samples_done < num_samples:
        print(f"Warning: only {samples_done} Banzhaf samples out of {num_samples}")

    # Average
    shapley = {sq: val / samples_done for sq, val in shapley_sums.items()}
    banzhaf = {sq: val / samples_done for sq, val in banzhaf_sums.items()}
    return shapley, banzhaf