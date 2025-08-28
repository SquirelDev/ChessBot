import multiprocessing
import os
import time
from queue import Empty

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
            # Get model's move
            board_tensor = np.expand_dims(board_to_tensor(board), axis=0)
            policy, value = chess_model.predict(board_tensor)

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

            board.push(move)

            # Get Stockfish's evaluation and best move
            stockfish.set_fen_position(board.fen())
            sf_eval = stockfish.get_evaluation()
            sf_best_move_uci = stockfish.get_best_move()

            if sf_best_move_uci:
                sf_best_move = chess.Move.from_uci(sf_best_move_uci)
                sf_best_move_index = move_to_index(sf_best_move)

                # Store the experience
                game_data.append((board_tensor.squeeze(0), sf_best_move_index, normalize_stockfish_eval(sf_eval)))

        # Send collected game data to the learner
        data_queue.put(game_data)
        print(f"Worker {worker_id}: Finished game {game_num + 1}. Data sent to learner.")

    print(f"Worker {worker_id}: Finished all games.")

def learner(data_queue, model_path):
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

    training_data = []

    while True: # Loop indefinitely to train
        try:
            # Get data from workers
            game_data = data_queue.get(timeout=30) # Wait for 30s
            training_data.extend(game_data)

            if len(training_data) >= config.BATCH_SIZE:
                print(f"Learner: Collected {len(training_data)} samples. Starting training.")

                # Sample a batch from the collected data
                indices = np.random.choice(len(training_data), config.BATCH_SIZE, replace=False)
                batch = [training_data[i] for i in indices]

                # Unzip the batch
                board_tensors, policy_targets, value_targets = zip(*batch)

                with tf.GradientTape() as tape:
                    # Forward pass
                    policy_preds, value_preds = chess_model(np.array(board_tensors))

                    # Calculate losses
                    policy_loss = loss_fn_policy(policy_targets, policy_preds)
                    value_loss = loss_fn_value(value_targets, value_preds)
                    total_loss = policy_loss + value_loss

                # Backward pass and optimization
                grads = tape.gradient(total_loss, chess_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, chess_model.trainable_variables))

                print(f"Learner: Training step completed. Total Loss: {total_loss.numpy():.4f}, Policy Loss: {policy_loss.numpy():.4f}, Value Loss: {value_loss.numpy():.4f}")

                # Save the updated model
                chess_model.save_weights(model_path)
                print(f"Learner: Model saved to {model_path}")

                # Clear the training data buffer
                training_data.clear()

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

    # Use a manager for the queue
    manager = multiprocessing.Manager()
    data_queue = manager.Queue()

    # Start the learner process
    learner_process = multiprocessing.Process(target=learner, args=(data_queue, model_path))
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
