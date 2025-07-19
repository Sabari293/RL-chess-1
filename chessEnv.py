import config
from re import A
import chess
from chess import Move
import numpy as np
import time
class ChessEnv:
    def __init__(self,fen:str=chess.STARTING_FEN):
        self.fen=fen
        self.reset()
    def __str__(self):
        return str(chess.Board(self.board))

    def step(self, action: Move) -> chess.Board:
        self.board.push(action)
        return self.board
    def reset(self):
        self.board=chess.Board(self.fen)
    @staticmethod
    def state_to_input(fen:str)->np.ndarray:
        board=chess.Board(fen)
        #next_move
        is_white=np.ones((8,8)) if board.turn else np.zeros((8,8))
        #castling
        castling = np.asarray([
            np.ones((8, 8)) if board.has_queenside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(
                chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_queenside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(
                chess.BLACK) else np.zeros((8, 8)),
        ])
        counter=np.ones((8,8)) if board.can_claim_fifty_moves else np.zeros((8,8))
        A=[]
        for color in chess.COLORS:
            for piece in chess.PIECE_TYPES:
                B=np.zeros((8,8))
                for i in list(board.pieces(piece,color)):
                    # row calculation: 7 - index/8 because we want to count from bottom left, not top left
                    B[7-int(i/7)][i%8]=True
                A.append(B)
        A=np.asarray(A)
        #enpassant
        en_passant = np.zeros((8, 8))
        if board.has_legal_en_passant():
            en_passant[7 - int(board.ep_square/8)][board.ep_square % 8] = True

        r = np.array([is_white, *castling,
                     counter, *A, en_passant]).reshape((1, *config.INPUT_SHAPE))
        # deleting memory
        del board
        return r.astype(bool)
    @staticmethod
    def estimate_winner(board:chess.Board)->int:
        score=0
        A={
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        for p in board.piece_map().values:
            if(p.color==chess.WHITE):
                score+=A[p.piece_type]
            else:
                score -= A[p.piece_type]
        if(np.abs(score)>5):
            if score>0:
                return 0.25
            else:
                return -0.25
        else:
            return 0
    @staticmethod
    def get_piece_amount(board:chess.Board)->int:
        return len(board.piece_map().values())
    


