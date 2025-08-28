import os
import time
from concurrent import futures

from . import config
from . import network
from . import replay_buffer
from . import self_play
from . import train

def main():
    """
    The main function to orchestrate the AlphaZero training pipeline.
    """
    # 1. Initialization
    # ------------------
    # Create the neural network
    current_network = network.create_model()

    # Initialize the replay buffer
    buffer = replay_buffer.ReplayBuffer()

    # Create directories for saving models and logs
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    print("AlphaZero Chess Engine: Initialization Complete.")
    print(f"Running on {config.NUM_WORKERS} parallel workers.")

    # 2. Main Training Loop
    # ---------------------
    # Save the initial model weights
    latest_model_path = os.path.join(config.CHECKPOINT_DIR, "latest_model.h5")
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
        if len(buffer) < config.BATCH_SIZE:
            print("Skipping training phase: not enough data in replay buffer.")
            continue

        start_time = time.time()
        print("Starting training phase...")

        # Run multiple training steps for each self-play phase
        # This ratio is a key hyperparameter.
        for _ in range(len(buffer) // config.BATCH_SIZE):
            loss_metrics = train.train_step(current_network, buffer)

        if loss_metrics:
            print(f"Training finished. Last batch loss: {loss_metrics['total_loss']:.4f}")

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
    # Note: Multiprocessing with TensorFlow can have issues with some CUDA
    # versions and setups. It's recommended to run this from the command line.
    main()
