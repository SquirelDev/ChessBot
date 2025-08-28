# AlphaZero-style Chess Engine

This project is an advanced, high-performance implementation of a chess engine based on the principles of the AlphaZero paper. It learns to play chess from scratch through a process of self-play and reinforcement learning, guided by a Monte Carlo Tree Search (MCTS) algorithm.

This system is designed to be run on powerful hardware (ideally a strong multi-core CPU and a high-end NVIDIA GPU) for extended periods to train a competitive chess model.

## Design Philosophy: Pure Self-Play

A key feature of this advanced architecture is that **it does not use Stockfish** or any other chess engine during its training. This is a deliberate design choice based on the success of systems like AlphaZero.

-   **Previous Approach (Imitation)**: The simpler engine built previously used Stockfish as a "teacher." The goal was to imitate Stockfish's moves and evaluations. This is a good way to create a reasonably strong engine, but its potential is ultimately limited by the teacher's strength.
-   **This Approach (Self-Discovery)**: This engine learns entirely from self-play. By starting with only the rules of chess and playing millions of games against itself, it is free to discover novel strategies and patterns that may not be part of established chess theory or typical engine play. This "tabula rasa" (blank slate) learning is what gives it the potential to surpass existing engines.

## Core Architecture

The training pipeline is composed of several key components that run in parallel:

1.  **Neural Network (`network.py`)**: The "brain" of the engine. It's a deep Residual Network (ResNet) with two heads:
    -   A **policy head** that predicts the probability of playing each possible move.
    -   A **value head** that estimates the probability of winning from the current position.

2.  **Monte Carlo Tree Search (`mcts.py`)**: A powerful search algorithm that explores the game tree. Instead of just relying on the network's raw policy, MCTS runs thousands of simulations to find the best move. The network's policy and value predictions are used to make this search intelligent and efficient.

3.  **Self-Play Workers (`self_play.py`)**: Multiple parallel processes that do one thing: play games of chess against themselves. At each turn, a worker uses the MCTS algorithm (guided by the latest neural network) to choose a move.

4.  **Replay Buffer (`replay_buffer.py`)**: A large buffer that stores the data from all completed self-play games. Each data point consists of a board state, the improved policy calculated by MCTS, and the final outcome of the game.

5.  **Trainer (`train.py`)**: A central process that continuously samples batches of data from the replay buffer and uses it to train and improve the neural network.

6.  **Orchestrator (`main.py`)**: The main script that launches and coordinates all the self-play workers and the trainer, creating a continuous loop of self-play and learning.

## How to Use

### 1. Prerequisites

- Python 3.8 or newer.
- A powerful NVIDIA GPU with CUDA and cuDNN installed.
- A strong multi-core CPU.

### 2. Installation

**1. Clone the repository:**
```bash
git clone <repository-url>
cd <repository-directory>
```

**2. Install Python dependencies:**
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```
*Note: Depending on your system, you may need to install a specific version of TensorFlow that matches your CUDA installation (e.g., `pip install tensorflow==2.10`).*

### 3. Configuration

Key parameters for the training process can be adjusted in `az_chess/config.py`. Some important ones include:
- `RESIDUAL_BLOCKS`: The depth of the neural network.
- `MCTS_SIMULATIONS`: The number of MCTS simulations per move. Higher is stronger but slower.
- `NUM_WORKERS`: The number of parallel self-play processes. A good starting point is the number of CPU cores on your machine.

### 4. Running the Training

To start the full training pipeline, run the `main.py` script:
```bash
python -m az_chess.main
```
This will start the main orchestrator, which will in turn launch the self-play workers. The system will create a `checkpoints/` directory to save model weights.

#### Stopping and Resuming
The training process is designed to be stopped and resumed at any time.
- To **stop**, simply use `Ctrl+C` in the terminal where the script is running.
- To **resume**, just run the same command again: `python -m az_chess.main`. The script will automatically find the `latest_model.h5` checkpoint in the `checkpoints/` directory and continue from there.

### 5. Interactive Demonstration
For a step-by-step, interactive walkthrough of the training loop, you can use the provided Jupyter notebook.
```bash
jupyter notebook az_chess/train_notebook.ipynb
```
This notebook is for educational and debugging purposes and runs a much-simplified, single-threaded version of the pipeline.

## Project Structure
```
.
├── README.md
├── requirements.txt
└── az_chess/
    ├── __init__.py
    ├── config.py           # All configuration and hyperparameters
    ├── main.py             # Main orchestrator to run the training pipeline
    ├── mcts.py             # Monte Carlo Tree Search implementation
    ├── network.py          # ResNet neural network architecture
    ├── replay_buffer.py    # The experience replay buffer
    ├── self_play.py        # Self-play game generation logic
    ├── train.py            # The network training step logic
    └── train_notebook.ipynb # Notebook for demonstration
```
