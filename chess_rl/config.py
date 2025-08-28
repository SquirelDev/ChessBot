# -----------------------------------------------------------------------------
# Stockfish Configuration
# -----------------------------------------------------------------------------

# IMPORTANT: Please update this path to the location of your Stockfish executable.
# You can download Stockfish from: https://stockfishchess.org/download/
STOCKFISH_PATH = "/path/to/stockfish"

# Set the skill level of Stockfish (0-20). 20 is the maximum.
STOCKFISH_SKILL_LEVEL = 20

# Set the thinking time for Stockfish in milliseconds.
STOCKFISH_THINK_TIME = 100

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

# The number of parallel worker processes to use for generating games.
NUM_WORKERS = 4

# The number of games to play for training.
NUM_TRAINING_GAMES = 1000

# The learning rate for the optimizer.
LEARNING_RATE = 0.001

# The size of the batch for training the neural network.
BATCH_SIZE = 64

# The number of epochs to train on the collected data.
EPOCHS = 10

# The directory where the trained model will be saved.
MODEL_DIR = "trained_model"

# The filename for the saved model.
MODEL_FILENAME = "chess_model.h5"
