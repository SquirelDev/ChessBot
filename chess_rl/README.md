# Chess RL - A Reinforcement Learning Chess Engine

This project implements a chess engine that learns to play chess using an advanced reinforcement learning approach. The neural network is trained by playing games and learning from feedback provided by the Stockfish engine.

## How It Works

The project is built around an **Actor-Critic (A2C)** model with an **Experience Replay** buffer.

1.  **Neural Network (Actor-Critic)**: A Convolutional Neural Network (CNN) that acts as both an Actor and a Critic.
    -   **The Actor (Policy Head)**: Decides which move to play from a given position.
    -   **The Critic (Value Head)**: Evaluates the current position, predicting the likely outcome.

2.  **Training Process**:
    -   **Data Generation**: Multiple "worker" processes play games, generating a stream of game data. For each move, the worker stores the state, the model's value prediction for that state, the chosen action, and the "reward" (which is Stockfish's evaluation of the *next* state).
    -   **Experience Replay**: This game data is stored in a large replay buffer. Storing a long history of experiences and sampling from it randomly helps to stabilize the training process.
    -   **Learning (A2C)**: A central "learner" process samples batches of data from the replay buffer. It uses the Actor-Critic algorithm to update the network. The "advantage" (how much better the actual outcome was than the Critic's prediction) is used to teach the Actor which moves are good.

## Getting Started

Follow these steps to set up and run the project.

### 1. Prerequisites

- Python 3.7 or newer.
- The Stockfish chess engine.

### 2. Installation

**1. Clone the repository:**
```bash
git clone <repository-url>
cd <repository-directory>/chess_rl
```

**2. Install Python dependencies:**
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

**3. Download Stockfish:**
You need to download the Stockfish engine executable for your operating system from the official website:
[https://stockfishchess.org/download/](https://stockfishchess.org/download/)

After downloading, unzip the file and place the Stockfish executable in a known location on your system.

### 3. Configuration

Before running the training script, you must specify the path to the Stockfish executable.

1.  Open the `config.py` file.
2.  Find the `STOCKFISH_PATH` variable.
3.  Update its value to the full path of your Stockfish executable.

**Example:**
```python
# On Linux or macOS
STOCKFISH_PATH = "/home/user/stockfish/stockfish"

# On Windows
# STOCKFISH_PATH = "C:\\Users\\user\\stockfish\\stockfish.exe"
```

You can also adjust other training parameters in `config.py`, such as the number of workers, learning rate, etc.

### 4. Running the Training

To start the training process, run the `train.py` script from within the `chess_rl` directory:
```bash
python train.py
```

The script will start the learner and worker processes. The workers will begin playing games against Stockfish and sending data to the learner. The learner will then train the model and periodically save its weights to the `trained_model/` directory.

You will see output from the workers as they finish games and from the learner as it completes training steps.

## 5. Visualizing Training Progress

After you have run the training script for a while, a `training_log.csv` file will be created in the `trained_model/` directory. You can visualize the training progress by running the Jupyter notebook.

First, make sure you have Jupyter installed and running:
```bash
pip install notebook
jupyter notebook
```

Then, from the Jupyter interface in your browser, open the `visualize_training.ipynb` notebook and run the cells. This will generate plots showing the training losses over time.

## 6. Evaluating the Model

To evaluate your trained model's performance, you can use the `evaluate.py` script. This script will play a number of games between your model and a Stockfish opponent.

**Usage:**
```bash
python evaluate.py [OPTIONS]
```

**Options:**
- `--model-path`: Path to the saved model file (e.g., `trained_model/chess_model.h5`).
- `--skill-level`: Stockfish's skill level to play against (0-20, default: 5).
- `--num-games`: The number of games to play for the evaluation (default: 10).

**Example:**
```bash
python evaluate.py --skill-level 8 --num-games 20
```
This will play 20 games against Stockfish at skill level 8.

## 7. Interactive Training with a Notebook

For a more interactive way to experiment with the training process, you can use the `train_notebook.ipynb`. This notebook contains a simplified, single-threaded version of the training loop. It is great for debugging and understanding the core logic, but it is much slower than the main `train.py` script.

To use it, start Jupyter Notebook and open the `train_notebook.ipynb` file, then run the cells sequentially.

## Project Structure
```
chess_rl/
├── README.md                   # This file
├── config.py                   # Configuration for Stockfish path and hyperparameters
├── model.py                    # The neural network model definition
├── requirements.txt            # Python dependencies
├── train.py                    # The main training script
├── utils.py                    # Helper functions for board representation and move mapping
├── visualize_training.ipynb    # Jupyter notebook for visualizing training logs
├── train_notebook.ipynb        # Jupyter notebook for interactive training
└── evaluate.py                 # Script for evaluating the model against Stockfish
```
