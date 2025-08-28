import multiprocessing
import os
import time
from queue import Empty
import csv
import collections
import random

import chess
import numpy as np
import tensorflow as tf
from stockfish import Stockfish

import config
from model import create_chess_model, NUM_POSSIBLE_MOVES
from utils import board_to_tensor, move_to_index, index_to_move, PIECE_TO_PLANE

def normalize_stockfish_eval(evaluation):
    """
    Normalizes Stockfish's evaluation to a range of -1 to 1.
    Handles 'cp' (centipawn) and 'mate' evaluations.
    """
    if evaluation['type'] == 'cp':
        # Use tanh to squash the value between -1 and 1.
        # The scaling factor (e.g., 600) can be tuned.
        return np.tanh(evaluation['value'] / 600.0)
    elif evaluation['type'] == 'mate':
        # Mate in X moves is a strong win/loss.
        return 1.0 if evaluation['value'] > 0 else -1.0
    return 0.0

def worker(worker_id, data_queue, model_path):
    """
    The worker process that plays games against Stockfish to generate training data.
    """
    print(f"Worker {worker_id}: Starting.")

    # 1. Initialize Stockfish
    try:
        stockfish = Stockfish(path=config.STOCKFISH_PATH, parameters={"Skill Level": config.STOCKFISH_SKILL_LEVEL})
    except (FileNotFoundError, PermissionError) as e:
        print(f"Worker {worker_id}: ERROR - Stockfish executable not found or not executable at '{config.STOCKFISH_PATH}'.")
        print("Please download Stockfish from https://stockfishchess.org/download/ and update STOCKFISH_PATH in config.py.")
        return

    # 2. Load the model
    chess_model = create_chess_model()
    if os.path.exists(model_path):
        print(f"Worker {worker_id}: Loading model from {model_path}")
        chess_model.load_weights(model_path)

    # 3. Game generation loop
    for game_num in range(config.NUM_TRAINING_GAMES // config.NUM_WORKERS):
        board = chess.Board()
        game_data = []

        while not board.is_game_over(claim_draw=True):
            # Get model's move and value prediction
            board_tensor = np.expand_dims(board_to_tensor(board), axis=0)
            policy_logits, predicted_value = chess_model.predict(board_tensor, verbose=0)
            predicted_value = predicted_value[0][0] # Get scalar value

            # Apply softmax to get probabilities
            policy = tf.nn.softmax(policy_logits[0]).numpy()

            # Select a move from the policy
            legal_moves = list(board.legal_moves)
            legal_move_indices = [move_to_index(m) for m in legal_moves]

            # Mask illegal moves and re-normalize probabilities
            legal_policy = np.zeros_like(policy[0])
            legal_policy[legal_move_indices] = policy[0][legal_move_indices]

            if np.sum(legal_policy) > 0:
                legal_policy /= np.sum(legal_policy)
                move = np.random.choice(legal_moves, p=legal_policy[legal_move_indices])
            else:
                # If policy gives zero probability to all legal moves, choose a random one
                move = np.random.choice(legal_moves)

            # Store the state and the model's prediction *before* the move
            state_for_buffer = board_tensor.squeeze(0)
            value_for_buffer = predicted_value
            action_for_buffer = move_to_index(move)

            board.push(move)

            # Get Stockfish's evaluation of the *new* position as the reward
            stockfish.set_fen_position(board.fen())
            sf_eval = stockfish.get_evaluation()
            reward = normalize_stockfish_eval(sf_eval)

            # Store the experience for A2C: (state, model_value, action, reward)
            game_data.append((state_for_buffer, value_for_buffer, action_for_buffer, reward))

        # Send collected game data to the learner
        data_queue.put(game_data)
        print(f"Worker {worker_id}: Finished game {game_num + 1}. Data sent to learner.")

    print(f"Worker {worker_id}: Finished all games.")

def learner(data_queue, model_path, log_path):
    """
    The learner process that trains the neural network on data from the workers.
    """
    print("Learner: Starting.")

    # 1. Initialize the model and optimizer
    chess_model = create_chess_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    # Define loss functions for the two heads
    loss_fn_policy = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_fn_value = tf.keras.losses.MeanSquaredError()

    if os.path.exists(model_path):
        print(f"Learner: Loading model from {model_path}")
        chess_model.load_weights(model_path)

    # 2. Setup Experience Replay Buffer and Logging
    replay_buffer = collections.deque(maxlen=config.REPLAY_BUFFER_SIZE)
    log_file_exists = os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not log_file_exists:
            writer.writerow(['step', 'total_loss', 'policy_loss', 'value_loss', 'timestamp'])

    step = 0

    while True: # Loop indefinitely to train
        try:
            # Get data from workers and add to replay buffer
            game_data = data_queue.get(timeout=60) # Wait for 60s
            replay_buffer.extend(game_data)

            if len(replay_buffer) >= config.BATCH_SIZE:
                step += 1
                print(f"Learner: Buffer size: {len(replay_buffer)}. Starting training step {step}.")

                # Sample a batch from the replay buffer
                batch = random.sample(replay_buffer, config.BATCH_SIZE)

                # Unzip the batch for A2C: (state, predicted_value, action, reward)
                states, predicted_values, actions, rewards = zip(*batch)

                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards, dtype=np.float32)

                with tf.GradientTape() as tape:
                    # Forward pass
                    policy_preds, value_preds = chess_model(states)

                    # Squeeze value_preds to match dimensions of rewards
                    value_preds = tf.squeeze(value_preds)

                    # --- Actor-Critic Loss Calculation ---

                    # Calculate Advantage (A = R - V(s))
                    # R is the reward (from Stockfish's eval of the next state)
                    # V(s) is our model's value prediction for the current state.
                    # A high advantage means the action taken led to a better-than-expected outcome.
                    advantage = rewards - value_preds

                    # 1. Critic Loss (Value Loss)
                    # The critic learns to evaluate positions better by minimizing the difference
                    # between its prediction and the actual reward (Stockfish's evaluation).
                    value_loss = loss_fn_value(rewards, value_preds)

                    # 2. Actor Loss (Policy Loss)
                    # The actor learns to choose better moves.
                    # We calculate the cross-entropy loss for the action taken.
                    policy_loss_per_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=policy_preds, labels=actions
                    )
                    # We then weight this loss by the advantage.
                    # Actions with a high advantage are reinforced (loss is minimized).
                    # We use stop_gradient because we only want to train the actor here, not the critic.
                    policy_loss = policy_loss_per_action * tf.stop_gradient(advantage)

                    # 3. Total Loss
                    # We combine the policy and value losses. The value loss is often weighted
                    # to stabilize training.
                    total_loss = tf.reduce_mean(policy_loss + 0.5 * value_loss)

                # Backward pass and optimization
                grads = tape.gradient(total_loss, chess_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, chess_model.trainable_variables))

                total_loss_val = total_loss.numpy()
                policy_loss_val = policy_loss.numpy()
                value_loss_val = value_loss.numpy()

                print(f"Learner: Step {step} completed. Total Loss: {total_loss_val:.4f}, Policy Loss: {policy_loss_val:.4f}, Value Loss: {value_loss_val:.4f}")

                # Log metrics
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, total_loss_val, policy_loss_val, value_loss_val, time.time()])

                # Save the updated model
                chess_model.save_weights(model_path)
                print(f"Learner: Model saved to {model_path}")

        except Empty:
            print("Learner: Queue is empty. No data from workers. Stopping.")
            break
        except Exception as e:
            print(f"Learner: An error occurred: {e}")
            break

if __name__ == '__main__':
    # Ensure the model directory exists
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    model_path = os.path.join(config.MODEL_DIR, config.MODEL_FILENAME)
    log_path = os.path.join(config.MODEL_DIR, config.TRAINING_LOG_FILE)

    # Use a manager for the queue
    manager = multiprocessing.Manager()
    data_queue = manager.Queue()

    # Start the learner process
    learner_process = multiprocessing.Process(target=learner, args=(data_queue, model_path, log_path))
    learner_process.start()

    # Start worker processes
    worker_processes = []
    for i in range(config.NUM_WORKERS):
        p = multiprocessing.Process(target=worker, args=(i + 1, data_queue, model_path))
        worker_processes.append(p)
        p.start()
        time.sleep(1) # Stagger worker starts

    # Wait for all workers to finish
    for p in worker_processes:
        p.join()

    print("Main: All workers have finished.")

    # Allow the learner to process any remaining data
    time.sleep(10)

    # Terminate the learner
    if learner_process.is_alive():
        learner_process.terminate()
        learner_process.join()

    print("Main: Training finished.")
