#mcts.py
import chess
import chess.pgn
from chessEnv import ChessEnv
from node import Node
from edge import Edge
import numpy as np
import time
from tqdm import tqdm
import utils
import threading
import tensorflow as tf
from mapper import Mapping

class MCTS:
    def __init__(self,agent:str="Agent",state:str=chess.STARTING_FEN,stochastic=False):
        self.root = Node(state=state)
        self.game_path: list[Edge] = []
        self.cur_board: chess.Board = None
        self.agent = agent
        self.stochastic = stochastic
    def run_simulations(self,n:int)->None:
        for _ in tqdm(range(n)):
            self.game_path=[]
            leaf=self.select_child(self.root)
            leaf.N+=1
            leaf=self.expand(leaf)
            leaf=self.backpropagate(leaf,leaf.value)
    def backpropagate(self,end_node:Node,value:float)->Node:
        for e in self.game_path:
            e.in_node.N+=1
            e.N+=1
            e.W+=value
    def select_child(self,node:Node)->Node:
        while not node.is_leaf():
            if not len(node.edges):
                return node
            noise = [1 for _ in range(len(node.edges))]
            if self.stochastic and node == self.root:
                noise = np.random.dirichlet([0.3]*len(node.edges))
            best_edge = None
            best_score = -np.inf                
            for i, e in enumerate(node.edges):
                score=e.ucb(noise[i])
                if score > best_score:
                    best_score = score
                    best_edge = e

            if best_edge is None:
                raise Exception("No edge found")
            node = best_edge.out_node
            self.game_path.append(best_edge)
        return node
    def map_valid_move(self,move:chess.Move)->None:
        from_square =move.from_square
        to_square=move.to_square
        plane_index:int=None
        p=self.cur_board.piece_at(from_square)
        direction = None
        if p is None:
            raise Exception(f"No piece at {from_square}")
        if move.promotion and move.promotion!=chess.QUEEN:
            piece_type,direction=Mapping.get_underpromotion_move(move.promotion,from_square,to_square)
            plane_index=Mapping.mapper[piece_type][1-direction]
        else:
            
            if p.piece_type == chess.KNIGHT:
                direction = Mapping.get_knight_move(from_square, to_square)
                plane_index = Mapping.mapper[direction]
            else:
                direction, distance = Mapping.get_queenlike_move(
                    from_square, to_square)
                plane_index = Mapping.mapper[direction][np.abs(distance)-1]
        row = from_square % 8
        col = 7 - (from_square // 8)
        self.outputs.append((move, plane_index, row, col))
    def probabilities_to_actions(self,p:list,board:str)->dict:
        p=p.reshape(73,8,8)
        actions={}
        self.cur_board=chess.Board(board)
        valid_moves=self.cur_board.generate_legal_moves()
        self.outputs=[]
        t=[]
        while True:
            try:
                move=next(valid_moves)
            except StopIteration:
                break
            thread=threading.Thread(target=self.map_valid_move,args=(move,))
            t.append(thread)
            thread.start()
        for thread in t:
            thread.join()
        for move,plane_index,col,row in self.outputs:
            actions[move.uci()]=p[plane_index][col][row]
        return actions
    def expand(self,leaf:Node)->Node:
        board=chess.Board(leaf.state)
        actions=list(board.generate_legal_moves())
        if not len(actions):
            assert board.is_game_over(),"Game is not over,but there is no possible move"
            outcome=board.outcome(claim_draw=True)
            if outcome is None:
                leaf.value=0
            else:
                leaf.value=1 if outcome.winner==chess.WHITE else -1
            return leaf
        input_state=ChessEnv.state_to_input(leaf.state)
        p,v=self.agent.predict(input_state)
        actions = self.probabilities_to_actions(p, leaf.state)
        leaf.value = v
        for uci_str, prob in actions.items():
            move = chess.Move.from_uci(uci_str)  
            new_state = leaf.step(move)
            leaf.add_child(Node(new_state), move, prob)

        return leaf
    def get_policy_distribution(self) -> dict:
        
        visits = np.array([edge.N for edge in self.root.edges])
        total = np.sum(visits)
        if total == 0:
            # fallback: uniform
            return {edge.move.uci(): 1/len(self.root.edges) for edge in self.root.edges}
        return {
            edge.move.uci(): edge.N / total
            for edge in self.root.edges
        }

    def best_move(self) -> chess.Move:
        if not self.root.edges:
            raise Exception("No edges to choose from.")
        return max(self.root.edges, key=lambda e: e.N).move
    def get_policy(self, board: chess.Board, simulations: int = 100) -> dict[str, float]:
        self.root = Node(state=board.fen())
        self.cur_board = board
        self.run_simulations(simulations)

        visits = np.array([edge.N for edge in self.root.edges])
        total = np.sum(visits)

        if total == 0 or not self.root.edges:
            legal_moves = list(board.legal_moves)
            return {move.uci(): 1 / len(legal_moves) for move in legal_moves}

        return {
            edge.move.uci(): edge.N / total
            for edge in self.root.edges
        }

