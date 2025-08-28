import tensorflow as tf
from tensorflow.keras import losses, optimizers

from . import config
from . import replay_buffer

def train_step(network: tf.keras.Model, buffer: replay_buffer.ReplayBuffer):
    """
    Performs a single training step on the neural network.
    """
    # 1. Sample a batch from the replay buffer
    batch = buffer.sample_batch(config.BATCH_SIZE)
    if batch is None:
        print("Trainer: Replay buffer does not have enough samples yet. Skipping training step.")
        return None

    states, policy_targets, value_targets = zip(*batch)

    states = tf.convert_to_tensor(states, dtype=tf.float32)
    policy_targets = tf.convert_to_tensor(policy_targets, dtype=tf.float32)
    value_targets = tf.convert_to_tensor(value_targets, dtype=tf.float32)

    # 2. Define loss functions and optimizer
    policy_loss_fn = losses.CategoricalCrossentropy(from_logits=True)
    value_loss_fn = losses.MeanSquaredError()
    optimizer = optimizers.Adam(learning_rate=config.LEARNING_RATE)

    # 3. Perform the training step
    with tf.GradientTape() as tape:
        # Forward pass
        policy_logits, value_preds = network(states, training=True)

        # Squeeze value_preds to match dimensions of value_targets
        value_preds = tf.squeeze(value_preds)

        # Calculate losses
        policy_loss = policy_loss_fn(policy_targets, policy_logits)
        value_loss = value_loss_fn(value_targets, value_preds)

        # Total loss is the sum of policy and value losses.
        # The AlphaZero paper does not mention weighting them differently.
        total_loss = policy_loss + value_loss

    # Backward pass and optimization
    grads = tape.gradient(total_loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    # Return loss values for logging
    return {
        "total_loss": total_loss.numpy(),
        "policy_loss": policy_loss.numpy(),
        "value_loss": value_loss.numpy()
    }
