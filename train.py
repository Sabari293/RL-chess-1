#train.py
import argparse
import os
import time
from typing import Tuple
import chess
import numpy as np
from chessEnv import ChessEnv
import config
import tensorflow as tf
from tensorflow.keras.models import Model,load_model, save_model
import pandas as pd
import uuid
import utils
from tqdm import tqdm
from datetime import datetime

class Trainer:
    def __init__(self,model:Model):
        self.model=model
        self.batch_size=config.BATCH_SIZE
    def sample_batch(self,data):
        if(self.batch_size>len(data)):
            return data
        else:
            np.random.shuffle(data)
            return data[:self.batch_size]
    def split_Xy(self,data)->Tuple[np.ndarray,np.ndarray]:
        X=np.array([ChessEnv.state_to_input(i[0])[0] for i in data])
        y_probs=[]
        y_values=[]
        for positions in data:
            board=chess.Board(positions[0])
            moves=utils.moves_to_output_vector(positions[1],board)
            y_probs.append(moves)
            y_values.append(positions[2])
        return X,(np.array(y_probs).reshape(len(y_probs),4672),np.array(y_values))
    def train_all_data(self,data):
        h=[]
        np.random.shuffle(data)
        print("Splitting data into labels and target...")
        X, y = self.split_Xy(data)
        print("Training batches...")
        for part in tqdm(range(len(X)//self.batch_size)):
            start = part * self.batch_size
            end = start + self.batch_size
            losses = self.train_batch(X[start:end], y[0][start:end], y[1][start:end])
            history.append(losses)
        return history
    def train_random_batches(self, data):
        history = []
        X, (y_probs, y_value) = self.split_Xy(data)
        for _ in tqdm(range(2*max(5, len(data) // self.batch_size))):
            indexes = np.random.choice(len(data), size=self.batch_size, replace=True)
            # only select X values with these indexes
            X_batch = X[indexes]
            y_probs_batch = y_probs[indexes]
            y_value_batch = y_value[indexes]
            
            losses = self.train_batch(X_batch, y_probs_batch, y_value_batch)
            history.append(losses)
        return history

    def train_batch(self, X, y_probs, y_value):
        return self.model.train_on_batch(x=X, y={
                "policy_head": y_probs,
                "value_head": y_value
            }, return_dict=True)


    def save_model(self):
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)
        path = f"{config.MODEL_FOLDER}/model-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.h5"
        save_model(self.model, path)
        print(f"Model trained. Saved model to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--model', type=str, help='The model to train')
    parser.add_argument('--data-folder', type=str, help='The data folder to train on')
    args = parser.parse_args()
    args = vars(args)
    model = tf.keras.models.load_model(args["model"])
    from keras.optimizers import Adam
    from keras.losses import CategoricalCrossentropy, MeanSquaredError

    model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss={
        "policy_head": CategoricalCrossentropy(from_logits=True),
        "value_head": MeanSquaredError()
    },
    metrics={
        "policy_head": "accuracy",
        "value_head": "mse"
    }
)

    trainer = Trainer(model=model)

    folder = args['data_folder']
    files = os.listdir(folder + "/")
    total = 0
    white_wins = 0
    black_wins = 0
    draws = 0

    print(f"Training on data from: {folder}")
    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(folder, file)
            print(f"\n Loading {file_path}")
            data = np.load(file_path, allow_pickle=True)

            total += len(data)
            white_wins += np.sum(data[:, 2] > 0)
            black_wins += np.sum(data[:, 2] < 0)
            draws += np.sum(data[:, 2] == 0)

            print(f"Training on {len(data)} positions...")
            trainer.train_random_batches(data)

            del data
            tf.keras.backend.clear_session()

    print("\n Training complete.")
    print(f"Total positions: {total}")
    print(f"White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
    trainer.save_model()