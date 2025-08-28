# Chess RL - A Reinforcement Learning Chess Engine

This project is a simple implementation of a chess engine that learns to play chess using a reinforcement learning approach. The neural network is trained by playing against the Stockfish engine and learning from its evaluations.

## How It Works

The project consists of two main components:
1.  **A Neural Network**: A Convolutional Neural Network (CNN) built with TensorFlow/Keras that takes a board position and outputs a policy (the best move to play) and a value (the evaluation of the position).
2.  **A Training Script**: A Python script that uses multiprocessing to parallelize the training process. Multiple "worker" processes play games against Stockfish to generate data, and a central "learner" process uses this data to train the neural network.

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

## Project Structure
```
chess_rl/
├── README.md           # This file
├── config.py           # Configuration for Stockfish path and hyperparameters
├── model.py            # The neural network model definition
├── requirements.txt    # Python dependencies
├── train.py            # The main training script
└── utils.py            # Helper functions for board representation and move mapping
```
