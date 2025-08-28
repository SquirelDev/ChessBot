import chess
import numpy as np

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

def _generate_uci_moves():
    """Generates a list of all possible UCI moves."""
    moves = []
    squares = [chess.square_name(s) for s in chess.SQUARES]
    for from_sq in squares:
        for to_sq in squares:
            if from_sq == to_sq:
                continue
            # Check for pawn promotions
            is_promotion = (('7' in from_sq and '8' in to_sq) or ('2' in from_sq and '1' in to_sq))
            if is_promotion:
                for piece in ['q', 'r', 'b', 'n']:
                    moves.append(f"{from_sq}{to_sq}{piece}")
            else:
                moves.append(f"{from_sq}{to_sq}")
    return moves

ALL_UCI_MOVES = _generate_uci_moves()
MOVE_TO_INDEX = {move: i for i, move in enumerate(ALL_UCI_MOVES)}
INDEX_TO_MOVE_UCI = {i: move for i, move in enumerate(ALL_UCI_MOVES)}
NUM_POSSIBLE_MOVES = len(ALL_UCI_MOVES)

def board_to_tensor(board: chess.Board) -> np.ndarray:
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
    return tensor

def move_to_index(move: chess.Move) -> int:
    """Converts a chess.Move object to its corresponding index."""
    return MOVE_TO_INDEX[move.uci()]

def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """Converts an index back to a chess.Move object."""
    uci_move = INDEX_TO_MOVE_UCI.get(index)
    if uci_move:
        try:
            return chess.Move.from_uci(uci_move)
        except chess.InvalidMoveError:
            return None
    return None

if __name__ == '__main__':
    print(f"Total number of possible UCI moves: {NUM_POSSIBLE_MOVES}")
    b = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move)
    print(f"Move 'e2e4' has index: {idx}")
    retrieved_move = index_to_move(idx, b)
    print(f"Index {idx} maps back to move: {retrieved_move.uci()}")
