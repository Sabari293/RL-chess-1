import os
import time
import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model,Model
from rlmodelbuilder import RLModelBuilder
from mcts import MCTS
import local_prediction

class Agent:
    def __init__(self,model_path:str,state=chess.STARTING_FEN):
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.mcts=MCTS(self,state=state)
    def build_model(self)->Model:
        model_builder=RLModelBuilder((8,8,19),(8*8*73,1))
        return model_builder.build_model()
    def run_simulations(self,N:int=1):
        self.mcts.run_simulations(N)
    def save_model(self,timestamped:bool=False):
        model_dir=os.environ.get("MODEL_FOLDER",".models")
        os.makedirs(model_dir,exist_ok=True)
        file=f"model-{time.time()}.h5" if timestamped else "model.h5"
        self.model.save(os.path.join(model_dir,file))
    def predict(self,data:np.ndarray):
        #this uses local model
        p, v = local_prediction.predict_local(self.model, data)
        return p.numpy(), v[0][0]