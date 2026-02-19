import chess
import numpy as np 

"""
There are 6 piece types:
    Pawn,Knight,Bishop,Rook,Queen,King
    Each can be:
        White or Black  so -> 6*2=12
    Map each piece 0 -> 11 for all 12 peices

Mapping each type of piece to an 8 by 8 representation of the board or a plane will give:
12 by 8 by 8 where a one or zero on each plane will indicate where the piece is to the model.

Example of a white pawn on e2 in i.e in the 0th plane:
    Each plane will be represented like so:
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0   <- white pawn at e2 
    0 0 0 0 0 0 0 0

Now to make neural nets and math easy to work with make a plane for the following:
    - Whose turn is it? white=1, black=0
        12 All 1's means whites turn all 0's is black's turn 
    - 2 planes for Queen and King side castling for both black and white ? TOTAL=4 planes i.e
        13	White can castle kingside
        14	White can castle queenside
        15	Black can castle kingside
        16	Black can castle queenside
    - En passant? which pawn has en passant is represented by a 1 else 0
        17 Which pawn has en passant is represented by a 1 else 0

    Hence total size of tensor for the chess board/state is:
    18 by 8 by 8 or (18,8,8)
"""

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Returns (18, 8, 8) float32 tensor.
    Planes:
      0-11: pieces
      12: side to move (all 1s if white to move else 0s)
      13-16: castling rights (WK,WQ,BK,BQ)
      17: en passant square (single 1 at ep square if exists)
    """
    x = np.zeros((18, 8, 8), dtype=np.float32)

    # pieces
    for sq, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        r = 7 - chess.square_rank(sq)  # put White's perspective at bottom
        f = chess.square_file(sq)
        x[plane, r, f] = 1.0

    # side to move
    if board.turn == chess.WHITE:
        x[12, :, :] = 1.0

    # castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        x[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        x[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        x[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        x[16, :, :] = 1.0

    # en passant
    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        f = chess.square_file(board.ep_square)
        x[17, r, f] = 1.0

    return x


# def move_features(board: chess.Board, move: chess.Move) -> np.ndarray:
#     """
#     Small feature vector for a move.
#     """
#     feats = []
#
#     # from/to normalized to [0,1]
#     feats.append(move.from_square / 63.0)
#     feats.append(move.to_square / 63.0)
#
#     # promotion: none=0, n=1, b=2, r=3, q=4 (normalized)
#     promo_map = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
#     feats.append(promo_map.get(move.promotion, 0) / 4.0)
#
#     # capture?
#     feats.append(1.0 if board.is_capture(move) else 0.0)
#
#     # castling?
#     feats.append(1.0 if board.is_castling(move) else 0.0)
#
#     # en passant?
#     feats.append(1.0 if board.is_en_passant(move) else 0.0)
#
#     # gives check? (need push/pop)
#     board.push(move)
#     feats.append(1.0 if board.is_check() else 0.0)
#     board.pop()
#
#     return np.array(feats, dtype=np.float32)  # shape (7,)
#
