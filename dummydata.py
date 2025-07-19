import chess
import numpy as np
import os
from datetime import datetime
import random

def generate_fake_game_data(num_positions=1):
    all_data = []

    for _ in range(num_positions):
        board = chess.Board()
        game_data = []

        while not board.is_game_over():
            fen = board.fen()
            legal_moves = list(board.legal_moves)
            chosen_move = random.choice(legal_moves)
            policy = {chosen_move.uci(): 1.0}
            game_data.append((fen, policy))  
            board.push(chosen_move)
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            result = 0 
        else:
            result = 1 if outcome.winner == chess.WHITE else -1
        for fen, policy in game_data:
            all_data.append((fen, policy, result))

    return all_data

if __name__ == "__main__":
    os.makedirs("data/selfplay", exist_ok=True)
    data = generate_fake_game_data(11)
    data = np.array(data, dtype=object)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data/selfplay/selfplay-{timestamp}.npy"
    np.save(filename, data)
    print(f"Saved dummy self-play data to {filename}")
