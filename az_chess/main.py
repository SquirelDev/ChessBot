import os
import time
from concurrent import futures
import multiprocessing

from . import config
from . import network
from . import replay_buffer
from . import self_play
from . import train
from . import utils

def main():
    """
    The main function to orchestrate the AlphaZero training pipeline.
    """
    utils.setup_gpu()

    # 1. Initialization
    # ------------------
    # Create the neural network
    current_network = network.create_model()

    # Initialize the replay buffer
    buffer = replay_buffer.ReplayBuffer()

    # Create directories for saving models and logs
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    # Check for the latest checkpoint to resume training
    latest_model_path = os.path.join(config.CHECKPOINT_DIR, "latest_model.h5")
    if os.path.exists(latest_model_path):
        print(f"Resuming training from checkpoint: {latest_model_path}")
        current_network.load_weights(latest_model_path)
    else:
        print("Starting new training session.")

    print("AlphaZero Chess Engine: Initialization Complete.")
    print(f"Running on {config.NUM_WORKERS} parallel workers.")

    # 2. Main Training Loop
    # ---------------------
    # Path for the latest model weights, used by workers
    latest_model_path = os.path.join(config.CHECKPOINT_DIR, "latest_model.h5")
    # Save the initial weights if starting a new session
    if not os.path.exists(latest_model_path):
        current_network.save_weights(latest_model_path)

    for step in range(1, config.TRAINING_STEPS + 1):
        print(f"\n--- Training Step {step}/{config.TRAINING_STEPS} ---")

        # --- Self-Play Phase ---
        start_time = time.time()
        print("Starting self-play phase...")

        # Use a process pool to run games in parallel
        with futures.ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
            # Each future will run one game of self-play using the latest model weights
            tasks = [executor.submit(self_play.run_game, latest_model_path) for _ in range(config.NUM_WORKERS)]

            # Collect results as they complete
            for future in futures.as_completed(tasks):
                game_hist = future.result()
                formatted_history = self_play.format_game_history_for_buffer(game_hist)
                buffer.save_game(formatted_history)

        self_play_duration = time.time() - start_time
        print(f"Self-play phase finished in {self_play_duration:.2f}s. Replay buffer size: {len(buffer)}")

        # --- Training Phase ---
        start_time = time.time()
        print("Starting training phase...")

        num_batches = len(buffer) // config.BATCH_SIZE
        if num_batches == 0:
            print("  Skipping training: not enough new data to form a full batch.")
            continue

        total_loss_sum = 0
        num_trained_batches = 0

        for i in range(num_batches):
            loss_metrics = train.train_step(current_network, buffer)
            if loss_metrics:
                total_loss_sum += loss_metrics['total_loss']
                num_trained_batches += 1
                # Print progress on a single line, `\r` returns the carriage
                print(f"  Training batch {i + 1}/{num_batches}... Loss: {loss_metrics['total_loss']:.4f}", end='\r')

        # Print a newline to move on from the progress line
        print()

        if num_trained_batches > 0:
            average_loss = total_loss_sum / num_trained_batches
            print(f"  Training finished. Average loss for this phase: {average_loss:.4f}")

        train_duration = time.time() - start_time
        print(f"Training phase finished in {train_duration:.2f}s.")

        # --- Save Checkpoint ---
        # Save the latest version for the next generation of workers
        current_network.save_weights(latest_model_path)

        # Also save a periodic checkpoint
        if step % 10 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_step_{step}.h5")
            current_network.save_weights(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("\n--- Training Finished ---")
    final_model_path = os.path.join(config.CHECKPOINT_DIR, "final_model.h5")
    current_network.save_weights(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    # Set the start method to 'spawn' to avoid CUDA initialization issues
    # in child processes. This is crucial for TensorFlow + multiprocessing.
    # 'force=True' is used to override the default if it has already been set.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # The context can only be set once.

    main()
