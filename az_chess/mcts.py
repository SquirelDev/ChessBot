import math
import numpy as np
import tensorflow as tf

from . import config
from . import utils

class Node:
    """
    A node in the Monte Carlo Tree Search. Represents a single board state.
    """
    def __init__(self, prior: float, parent=None, action: int = None):
        self.parent = parent
        self.action = action  # The action that led to this node from the parent
        self.children = {}  # A map from action to Node

        self.visit_count = 0
        self.total_action_value = 0.0  # Q-value
        self.prior = prior  # P-value, from the network's policy head

    @property
    def q_value(self) -> float:
        """Calculates the average action value (Q) for this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_action_value / self.visit_count

    def select_child(self):
        """
        Selects the child with the highest PUCT (Polynomial Upper Confidence Trees) score.
        """
        best_score = -np.inf
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.q_value + config.MCTS_C_PUCT * child.prior * \
                    (math.sqrt(self.visit_count) / (1 + child.visit_count))
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy_probs: np.ndarray, board):
        """
        Expands this node by creating children for all legal moves from the current state.
        """
        for move in board.legal_moves:
            action_idx = utils.MOVE_TO_INDEX.get(move.uci())
            if action_idx is not None:
                # The prior probability for this action from the network
                prior_p = policy_probs[action_idx]
                self.children[action_idx] = Node(prior=prior_p, parent=self, action=action_idx)

    def is_expanded(self) -> bool:
        """Checks if this node has been expanded."""
        return len(self.children) > 0


def run_mcts(board, network, simulations: int = config.MCTS_SIMULATIONS):
    """
    Runs the main MCTS loop for a given board state.
    """
    root = Node(prior=0.0)

    # Get initial policy and value from the network for the root state
    board_tensor = np.expand_dims(utils.board_to_tensor(board), axis=0)
    policy_logits, value = network.predict(board_tensor, verbose=0)
    policy_probs = tf.nn.softmax(policy_logits[0]).numpy()

    # Add Dirichlet noise for exploration at the root node
    if config.DIRICHLET_ALPHA > 0:
        legal_moves_mask = np.zeros(config.NUM_POSSIBLE_MOVES, dtype=np.float32)
        for move in board.legal_moves:
            legal_moves_mask[utils.MOVE_TO_INDEX[move.uci()]] = 1

        dirichlet_noise = np.random.dirichlet([config.DIRICHLET_ALPHA] * int(np.sum(legal_moves_mask)))
        noisy_policy = policy_probs[legal_moves_mask == 1] * (1 - config.DIRICHLET_EPSILON) + \
                       dirichlet_noise * config.DIRICHLET_EPSILON

        policy_probs[legal_moves_mask == 1] = noisy_policy

    root.expand(policy_probs, board)

    for _ in range(simulations):
        node = root
        search_path = [node]
        current_board = board.copy()

        # 1. Selection: Traverse the tree until a leaf node is found
        while node.is_expanded():
            action, node = node.select_child()
            if node is None: # Should not happen if there are legal moves
                break
            current_board.push(utils.INDEX_TO_MOVE_UCI[action])
            search_path.append(node)

        if node is None: continue # Path ended unexpectedly

        # 2. Expansion & Evaluation
        # Now at a leaf node. Get value from the network.
        if not current_board.is_game_over(claim_draw=True):
            board_tensor = np.expand_dims(utils.board_to_tensor(current_board), axis=0)
            policy_logits, value = network.predict(board_tensor, verbose=0)
            policy_probs = tf.nn.softmax(policy_logits[0]).numpy()
            value = value[0][0]
            # Expand the leaf node
            node.expand(policy_probs, current_board)
        else:
            # Game is over, the value is determined by the result
            if current_board.is_checkmate():
                # The player whose turn it is has been checkmated
                value = -1.0
            else: # Draw
                value = 0.0

        # 3. Backpropagation: Update node statistics up the search path
        for node_in_path in reversed(search_path):
            node_in_path.visit_count += 1
            # The value is from the perspective of the player at that node's state.
            # We need to flip the sign of the value at each step up the tree.
            node_in_path.total_action_value += value
            value = -value # Flip value for the parent node

    return root
