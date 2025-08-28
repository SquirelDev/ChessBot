import argparse
import os

import chess
import numpy as np
from stockfish import Stockfish

import config
from model import create_chess_model
from utils import board_to_tensor, move_to_index

def get_model_move(model, board):
    """Gets the best legal move from the model's policy."""
    board_tensor = np.expand_dims(board_to_tensor(board), axis=0)
    policy, _ = model.predict(board_tensor, verbose=0)

    legal_moves = list(board.legal_moves)
    legal_move_indices = [move_to_index(m) for m in legal_moves]

    # Get probabilities for legal moves
    legal_policy = policy[0][legal_move_indices]

    # Choose the legal move with the highest probability
    best_move_index = np.argmax(legal_policy)
    return legal_moves[best_move_index]

def get_stockfish_move(stockfish, board):
    """Gets the best move from Stockfish."""
    stockfish.set_fen_position(board.fen())
    best_move_uci = stockfish.get_best_move()
    return chess.Move.from_uci(best_move_uci)

def play_game(model, stockfish, model_plays_white):
    """
    Plays a single game between the model and Stockfish.
    Returns 'win', 'loss', or 'draw' from the model's perspective.
    """
    board = chess.Board()

    while not board.is_game_over(claim_draw=True):
        if (board.turn == chess.WHITE and model_plays_white) or \
           (board.turn == chess.BLACK and not model_plays_white):
            move = get_model_move(model, board)
        else:
            move = get_stockfish_move(stockfish, board)

        board.push(move)

    result = board.result(claim_draw=True)

    if result == '1-0': # White won
        return 'win' if model_plays_white else 'loss'
    elif result == '0-1': # Black won
        return 'loss' if model_plays_white else 'win'
    else: # Draw
        return 'draw'

def main(args):
    # 1. Load Model
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    print(f"Loading model from '{args.model_path}'...")
    model = create_chess_model()
    model.load_weights(args.model_path)
    print("Model loaded successfully.")

    # 2. Initialize Stockfish
    try:
        stockfish = Stockfish(path=config.STOCKFISH_PATH, parameters={"Skill Level": args.skill_level})
    except (FileNotFoundError, PermissionError):
        print(f"Error: Stockfish executable not found at '{config.STOCKFISH_PATH}'.")
        return
    print(f"Stockfish initialized with skill level {args.skill_level}.")

    # 3. Run evaluation games
    print(f"Starting evaluation with {args.num_games} games...")
    results = {'win': 0, 'loss': 0, 'draw': 0}

    for i in range(args.num_games):
        model_plays_white = (i % 2 == 0)
        color = "White" if model_plays_white else "Black"
        print(f"--- Starting Game {i+1}/{args.num_games} (Model plays as {color}) ---")

        game_result = play_game(model, stockfish, model_plays_white)
        results[game_result] += 1

        print(f"Game {i+1} result: Model {game_result.upper()}")
        print(f"Current Score: Wins={results['win']}, Losses={results['loss']}, Draws={results['draw']}")

    # 4. Print final results
    print("\n--- Evaluation Finished ---")
    print(f"Opponent: Stockfish (Skill Level {args.skill_level})")
    print(f"Total Games: {args.num_games}")
    print(f"Final Score: Wins={results['win']}, Losses={results['loss']}, Draws={results['draw']}")
    win_rate = (results['win'] / args.num_games) * 100
    print(f"Model Win Rate: {win_rate:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained chess model against Stockfish.")
    parser.add_argument(
        '--model-path',
        type=str,
        default=os.path.join(config.MODEL_DIR, config.MODEL_FILENAME),
        help='Path to the trained model weights file (.h5).'
    )
    parser.add_argument(
        '--skill-level',
        type=int,
        default=5,
        help='Skill level for the Stockfish opponent (0-20).'
    )
    parser.add_argument(
        '--num-games',
        type=int,
        default=10,
        help='Number of games to play for the evaluation.'
    )

    args = parser.parse_args()
    main(args)
