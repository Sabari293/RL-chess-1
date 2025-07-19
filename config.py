#config.py
import os
from dotenv import load_dotenv
load_dotenv()
SIMULATIONS_PER_MOVE = int(os.environ.get("SIMULATIONS_PER_MOVE", 1000))
C_base=20000
C_init=2
n = 8  
amount_of_input_planes =  (2*6 + 1) + (1 + 4 + 1)
INPUT_SHAPE = (n, n, amount_of_input_planes)
queen_planes = 56
knight_planes = 8
underpromotion_planes = 9
amount_of_planes = queen_planes + knight_planes + underpromotion_planes
OUTPUT_SHAPE = (8*8*amount_of_planes, 1)
LEARNING_RATE = 0.02
CONVOLUTION_FILTERS = 256
AMOUNT_OF_RESIDUAL_BLOCKS = 19
MODEL_FOLDER = os.environ.get("MODEL_FOLDER" ,'./models')
BATCH_SIZE = 16
LOSS_PLOTS_FOLDER="./plots"
MEMORY_DIR = os.environ.get("MEMORY_FOLDER", "./memory")
MAX_REPLAY_MEMORY = 1000000
SOCKET_BUFFER_SIZE = 8192