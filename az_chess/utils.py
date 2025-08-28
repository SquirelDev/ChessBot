import chess
import numpy as np

# --- Move Representation ---
# This section handles the mapping between chess moves and the indices used
# by the neural network's policy head.

def _generate_uci_moves():
    """Generates a list of all possible UCI moves for a standard chess board."""
    moves = []
    squares = [chess.square_name(s) for s in chess.SQUARES]
    for from_sq in squares:
        for to_sq in squares:
            if from_sq == to_sq:
                continue

            # Add the standard move (e.g., 'e2e4')
            moves.append(f"{from_sq}{to_sq}")

            # Add promotion moves (e.g., 'e7e8q')
            is_white_promo = ('7' in from_sq and '8' in to_sq)
            is_black_promo = ('2' in from_sq and '1' in to_sq)

            if is_white_promo or is_black_promo:
                for piece in ['q', 'r', 'b', 'n']:
                    moves.append(f"{from_sq}{to_sq}{piece}")

    return list(dict.fromkeys(moves))

ALL_UCI_MOVES = _generate_uci_moves()
MOVE_TO_INDEX = {move: i for i, move in enumerate(ALL_UCI_MOVES)}
INDEX_TO_MOVE_UCI = {i: move for i, move in enumerate(ALL_UCI_MOVES)}
NUM_POSSIBLE_MOVES = len(ALL_UCI_MOVES)


# --- Board Representation ---
# This section handles the conversion of a python-chess board object into
# a tensor suitable for input to the neural network.

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Converts a chess.Board object to an 8x8x18 tensor representation."""
    tensor = np.zeros((8, 8, 18), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            tensor[chess.square_rank(square), chess.square_file(square), plane] = 1
    if board.has_kingside_castling_rights(chess.WHITE): tensor[:, :, 12] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[:, :, 13] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[:, :, 14] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[:, :, 15] = 1
    if board.ep_square:
        tensor[chess.square_rank(board.ep_square), chess.square_file(board.ep_square), 16] = 1
    if board.turn == chess.WHITE: tensor[:, :, 17] = 1
    else: tensor[:, :, 17] = 0 # Explicitly set to 0 for black's turn
    return tensor
