#edge.py
from chess import Move
import config
import math
import chess
class Edge:
    def __init__(self,in_node,out_node,action:Move,prior:float):
        self.in_node=in_node
        self.out_node=out_node
        self.action=action
        self.player_turn=self.in_node.state.split(" ")[1]=="w"
        self.N=0#No of times actions used
        self.W=0#total action value
        self.P=prior#prior probability of selecting actions
    def __eq__(self, edge: object) -> bool:
        if isinstance(edge, Edge):
            return self.action == edge.action and self.in_node.state == edge.in_node.state
        return NotImplemented

    def __str__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.ucb()}"
    def __repr__(self):
        return f"{self.action.uci()}: Q={self.W / self.N if self.N != 0 else 0}, N={self.N}, W={self.W}, P={self.P}, U = {self.ucb()}"
    def ucb(self,noise:float)->float:
        exp_rate=math.log(((1+self.in_node.N+config.C_base)/config.C_base))+config.C_init
        value=exp_rate*(math.sqrt((self.in_node.N)/(1+self.N)))*(self.P*noise)
        if self.in_node.turn == chess.WHITE:
            return self.W / (self.N + 1) + value 
        else:
            return -(self.W / (self.N + 1)) + value