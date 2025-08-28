import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from . import config

def _residual_block(input_tensor, filters):
    """A single residual block for the ResNet."""
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([input_tensor, x])
    x = layers.ReLU()(x)
    return x

def create_model():
    """
    Creates the full ResNet-based neural network for chess.
    This architecture is inspired by the AlphaZero paper.
    """
    # --- Input Layer ---
    input_tensor = layers.Input(shape=config.BOARD_SHAPE)

    # --- Convolutional Tower ---
    # The input tower prepares the board representation for the residual blocks.
    x = layers.Conv2D(config.CONV_FILTERS, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # --- Residual Body ---
    # A stack of residual blocks to learn deep features.
    for _ in range(config.RESIDUAL_BLOCKS):
        x = _residual_block(x, config.CONV_FILTERS)

    # --- Policy Head ---
    # Outputs a probability distribution over all possible moves.
    policy_conv = layers.Conv2D(2, (1, 1), kernel_regularizer=regularizers.l2(1e-4))(x)
    policy_bn = layers.BatchNormalization()(policy_conv)
    policy_relu = layers.ReLU()(policy_bn)
    policy_flat = layers.Flatten()(policy_relu)
    # Outputs raw logits, softmax will be applied in the loss function or during inference.
    policy_logits = layers.Dense(config.NUM_POSSIBLE_MOVES, name='policy', kernel_regularizer=regularizers.l2(1e-4))(policy_flat)

    # --- Value Head ---
    # Outputs a single scalar value (-1 to 1) estimating the game's outcome.
    value_conv = layers.Conv2D(1, (1, 1), kernel_regularizer=regularizers.l2(1e-4))(x)
    value_bn = layers.BatchNormalization()(value_conv)
    value_relu = layers.ReLU()(value_bn)
    value_flat = layers.Flatten()(value_relu)
    value_hidden = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(value_flat)
    value_output = layers.Dense(1, activation='tanh', name='value', kernel_regularizer=regularizers.l2(1e-4))(value_hidden)

    # --- Create and Compile Model ---
    model = models.Model(inputs=input_tensor, outputs=[policy_logits, value_output])

    return model

if __name__ == '__main__':
    # Example of creating the model and printing its summary
    # This helps verify the architecture.
    az_model = create_model()
    az_model.summary()
    # You can also save a plot of the model architecture
    # tf.keras.utils.plot_model(az_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
