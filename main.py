# main.py
import time
import numpy as np
import chess
from chessEnv import ChessEnv
from game import Game 
from agent import Agent
import argparse

class Main:
    def __init__(self, player: bool, model_path: str, selfplay: bool = False):
        self.selfplay = selfplay
        self.previous_moves = (None, None)

        if self.selfplay:
            self.white_agent = Agent(model_path=model_path)
            self.black_agent = Agent(model_path=model_path)
            self.game = Game(ChessEnv(), self.white_agent, self.black_agent)
        else:
            self.player = player
            self.opponent = Agent(model_path=model_path)
            if self.player:
                self.game = Game(ChessEnv(), None, self.opponent)
            else:
                self.game = Game(ChessEnv(), self.opponent, None)
            print("$" * 50)
            print(f"You play the {'white' if self.player else 'black'} pieces!")
            print("$" * 50)

        self.play_game()

    def play_game(self):
        self.game.reset()
        board = self.game.env.board
        winner = None

        while winner is None:
            print("\nCurrent board:\n")
            print(board)
            print()

            if self.selfplay:
                print("Agent is thinking...")
                self.previous_moves = self.game.play_move(
                    stochastic=False,
                    previous_moves=self.previous_moves,
                    save_moves=True
                )
            else:
                if self.player == self.game.turn:
                    self.get_player_move()
                    self.game.turn = not self.game.turn
                else:
                    print("Opponent is thinking...")
                    self.opponent_move()
                    print(f"Opponent played: {board.peek()}")

            if board.is_game_over():
                print("\nFinal board:\n")
                print(board)
                print()
                winner = Game.get_winner(board.result(claim_draw=True))
                print("White wins" if winner == 1 else "Black wins" if winner == -1 else "Draw")
                break

    def get_player_move(self):
        board = self.game.env.board
        while True:
            move_input = input("Enter your move (e.g., e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_input)
                if move in board.legal_moves:
                    board.push(move)
                    break
                else:
                    print("Illegal move. Try again.")
            except:
                print("Invalid format. Use UCI like e2e4.")

    def opponent_move(self):
        self.previous_moves = self.game.play_move(
            stochastic=False,
            previous_moves=self.previous_moves,
            save_moves=True
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terminal Chess vs AI or AI vs AI")
    parser.add_argument("--selfplay", action="store_true", help="Run AI vs AI selfplay")
    parser.add_argument("--player", type=str, default=None, choices=("white", "black"), help="Play as white or black. Optional.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to use")

    args = parser.parse_args()
    player = args.player.lower().strip() == 'white' if args.player else np.random.choice([True, False])
    
    m = Main(player, model_path=args.model, selfplay=args.selfplay)
