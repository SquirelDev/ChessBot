import tensorflow as tf
from tensorflow.keras import layers, models

from utils import NUM_POSSIBLE_MOVES

BOARD_SHAPE = (8, 8, 18)  # 8x8 board with 18 feature planes

def create_chess_model(input_shape=BOARD_SHAPE):
    """
    Creates a convolutional neural network for chess.
    The model has a shared body and two heads: one for the policy (move probabilities)
    and one for the value (position evaluation).
    """
    input_tensor = layers.Input(shape=input_shape)

    # Shared Body
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Policy Head
    # Outputs raw logits. Softmax will be applied in the loss function.
    policy_head = layers.Dense(NUM_POSSIBLE_MOVES, name='policy')(x)

    # Value Head
    value_head = layers.Dense(1, activation='tanh', name='value')(x)

    model = models.Model(inputs=input_tensor, outputs=[policy_head, value_head])
    return model

if __name__ == '__main__':
    chess_model = create_chess_model()
    chess_model.summary()
