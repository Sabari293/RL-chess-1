#game.py
import os
import time
from chessEnv import ChessEnv
from agent import Agent
import utils
import config
from chess.pgn import Game as ChessGame
from edge import Edge
from mcts import MCTS
import uuid
import pandas as pd
import numpy as np

class Game:
    def __init__(self, env: ChessEnv, white: Agent, black: Agent):
        #The Game class is used to play chess games between two agents.
        
        self.env = env
        self.white = white
        self.black = black

        self.memory = []

        self.reset()

    def reset(self):
        self.env.reset()
        self.turn = self.env.board.turn  
        self.memory = [[]]

    @staticmethod
    def get_winner(result: str) -> int:
        return 1 if result == "1-0" else - 1 if result == "0-1" else 0

    def play_move(self, stochastic: bool = True, previous_moves: tuple[Edge, Edge] = (None, None), save_moves=True) -> None:
        current_player = self.white if self.turn else self.black

        if previous_moves[0] is None or previous_moves[1] is None:

            current_player.mcts = MCTS(current_player, state=self.env.board.fen(), stochastic=stochastic)
        else:
            
            try:
                node = current_player.mcts.root.get_edge(previous_moves[0].action).output_node
                node = node.get_edge(previous_moves[1].action).output_node
                current_player.mcts.root = node
            except AttributeError:
                current_player.mcts = MCTS(current_player, state=self.env.board.fen(), stochastic=stochastic)
        current_player.mcts.run_simulations(n=config.SIMULATIONS_PER_MOVE)

        moves = current_player.mcts.root.edges

        if save_moves:
            self.save_to_memory(self.env.board.fen(), moves)

        sum_move_visits = sum(e.N for e in moves)
        probs = [e.N / sum_move_visits for e in moves]
        
        if stochastic:
            best_move = np.random.choice(moves, p=probs)
        else:
            best_move = moves[np.argmax(probs)]
        self.env.step(best_move.action)
        self.turn = not self.turn
        return (previous_moves[1], best_move)

    def save_to_memory(self, state, moves) -> None:
        sum_move_visits = sum(e.N for e in moves)
        search_probabilities = {
            e.action.uci(): e.N / sum_move_visits for e in moves}
        self.memory[-1].append((state, search_probabilities, None))

    