#node.py
import chess
from chess import Move
from edge import Edge
class Node:
    def __init__(self,state:str):
        self.state=state
        self.turn=chess.Board(state).turn
        self.value=0
        self.N=0 #visit count
        self.edges:list[Edge]=[]# contains edges connected to it
    def __eq__(self,node:object) -> bool:
        #checks if ntwoo nodes are equal,if states are equal
        if isinstance(node, Node): #checks if both are same datatype
            return self.state == node.state
        else:
            return NotImplemented
    def add_child(self,child,action:Move,prior:float)->Edge:#adds child which is result after the given move on the chess Board
        edge=Edge(in_node=self,out_node=child,action=action,prior=prior)
        self.edges.append(edge)
        return edge
    def get_all_children(self)->list:
        A=[]
        for e in self.edges:
            A.append(e.out_node)
            A.extend(e.out_node.get_all_children())
        return A
    def get_edge(self,action:Move)->Edge:
        for e in self.edges:
            if(e.action==action):
                return e
        return None
    def is_leaf(self)->bool:#checks if Node is a leaf.
        return self.N==0
    def is_game_over(self):
        return chess.Board(self.state).is_game_over()
    def step(self,action:Move):#This helps in changing the state of the chess Board after a move.
        board = chess.Board(self.state)
        board.push(action)
        new_state = board.fen()
        del board
        return new_state

