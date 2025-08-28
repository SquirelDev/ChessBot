import chess
import numpy as np
import tensorflow as tf

from . import config
from . import mcts
from . import utils

class GameHistory:
    """
    Stores the history of a single game for training.
    """
    def __init__(self):
        self.states = []
        self.mcts_policies = []
        self.outcome = 0

    def add_step(self, state, policy):
        self.states.append(state)
        self.mcts_policies.append(policy)

    def set_outcome(self, outcome):
        self.outcome = outcome

from . import network

def run_game(model_path: str):
    """
    Runs a single game of self-play and returns the game history.
    This function is designed to be called in a separate process.
    """
    # Load the latest version of the network for this game
    model = network.create_model()
    model.load_weights(model_path)

    game_history = GameHistory()
    board = chess.Board()

    while not board.is_game_over(claim_draw=True):
        # Run MCTS to get the improved policy and the best move
        root_node = mcts.run_mcts(board, model)

        # Create the MCTS-based policy target for training
        policy_target = np.zeros(config.NUM_POSSIBLE_MOVES, dtype=np.float32)
        total_visits = sum(child.visit_count for child in root_node.children.values())
        if total_visits > 0:
            for action, child in root_node.children.items():
                policy_target[action] = child.visit_count / total_visits

        # Store the current state and the MCTS policy
        state_tensor = utils.board_to_tensor(board)
        game_history.add_step(state_tensor, policy_target)

        # Select a move to play.
        # For the first few moves, sample from the visit counts to ensure variety.
        # Afterwards, play the most visited move.
        if len(game_history.states) < 15:
            move_idx = np.random.choice(
                list(root_node.children.keys()),
                p=[child.visit_count / total_visits for child in root_node.children.values()]
            )
        else:
            move_idx = max(root_node.children.items(), key=lambda item: item[1].visit_count)[0]

        move = utils.INDEX_TO_MOVE_UCI[move_idx]
        board.push(chess.Move.from_uci(move))

    # Game is over, determine the outcome
    result = board.result(claim_draw=True)
    if result == '1-0':
        outcome = 1.0 # White won
    elif result == '0-1':
        outcome = -1.0 # Black won
    else:
        outcome = 0.0 # Draw

    game_history.set_outcome(outcome)
    return game_history

def format_game_history_for_buffer(game_history: GameHistory):
    """
    Formats the game history into tuples suitable for the replay buffer.
    (state, policy_target, value_target)
    """
    training_samples = []
    current_player = 1 # Start with White
    for i in range(len(game_history.states)):
        # The value target is the final game outcome from the perspective of the current player
        value_target = game_history.outcome * current_player
        training_samples.append((
            game_history.states[i],
            game_history.mcts_policies[i],
            value_target
        ))
        current_player *= -1 # Switch player perspective
    return training_samples
