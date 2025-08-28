# -----------------------------------------------------------------------------
# Chess Board and Move Representation
# -----------------------------------------------------------------------------

# The input to the network is a stack of 8x8 planes.
# We use 18 planes to represent the board state:
# - 12 planes for piece positions (6 types for each color)
# - 4 planes for castling rights (WK, WQ, BK, BQ)
# - 1 plane for the en passant square
# - 1 plane for the side to move
BOARD_SHAPE = (8, 8, 18)

from . import utils

# The number of possible moves is determined by the move representation in utils.py
NUM_POSSIBLE_MOVES = utils.NUM_POSSIBLE_MOVES

# -----------------------------------------------------------------------------
# Neural Network Architecture
# -----------------------------------------------------------------------------

# Number of residual blocks in the network body.
# AlphaZero used 19 or 39. We'll start with a smaller number for faster training.
RESIDUAL_BLOCKS = 7

# Number of filters in the convolutional layers.
# AlphaZero used 256. We'll use a smaller number to fit on consumer GPUs.
CONV_FILTERS = 128

# -----------------------------------------------------------------------------
# MCTS Configuration
# -----------------------------------------------------------------------------

# Number of simulations to run per move during self-play.
MCTS_SIMULATIONS = 100

# Exploration constant (c_puct) in the PUCT formula.
# Controls the trade-off between exploiting promising moves and exploring less-visited ones.
MCTS_C_PUCT = 1.25

# Dirichlet noise to add to the root node's policy for exploration.
# This encourages the MCTS to explore a wider variety of opening moves.
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------

# The number of parallel self-play workers.
# Should be adjusted based on available CPU cores.
NUM_WORKERS = 8

# The maximum size of the replay buffer (number of game states).
REPLAY_BUFFER_SIZE = 100000

# The number of training steps to run.
TRAINING_STEPS = 10000

# The batch size for training the neural network.
BATCH_SIZE = 256

# The learning rate for the Adam optimizer.
LEARNING_RATE = 0.001

# Directory to save model checkpoints and logs.
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
