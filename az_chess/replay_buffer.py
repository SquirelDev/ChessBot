import collections
import random

from . import config

class ReplayBuffer:
    """
    A buffer to store game states and training targets for experience replay.
    """
    def __init__(self, buffer_size: int = config.REPLAY_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=buffer_size)

    def save_game(self, game_history):
        """
        Saves a completed game's history to the buffer.
        Each entry in game_history should be a tuple of:
        (board_tensor, mcts_policy, game_outcome)
        """
        self.buffer.extend(game_history)

    def sample_batch(self, batch_size: int = config.BATCH_SIZE):
        """
        Samples a random batch of experiences from the buffer.
        """
        if len(self.buffer) < batch_size:
            # Not enough data to form a full batch
            return None

        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# This part is for shared memory implementation, which is more complex.
# For now, we will pass data through queues, but this is a placeholder
# for a more advanced implementation.
# For true high-performance, the replay buffer would be a separate process
# managing shared memory to avoid data serialization costs.
# For this implementation, we will instantiate it within the main training loop coordinator.
