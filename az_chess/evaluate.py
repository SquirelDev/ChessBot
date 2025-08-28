import argparse
import os
import math
import chess
from stockfish import Stockfish

from . import config
from . import network
from . import mcts
from . import utils

# Approximate Elo ratings for Stockfish skill levels (0-20)
# Based on community estimates, not official figures.
STOCKFISH_SKILL_TO_ELO = {
    0: 1350, 1: 1400, 2: 1450, 3: 1500, 4: 1550,
    5: 1600, 6: 1650, 7: 1700, 8: 1800, 9: 1900,
    10: 2000, 11: 2100, 12: 2200, 13: 2300, 14: 2400,
    15: 2500, 16: 2600, 17: 2700, 18: 2800, 19: 2900,
    20: 3000
}

def calculate_elo(score, opponent_elo, num_games):
    """
    Calculates the estimated Elo rating based on a match score.
    """
    if num_games == 0:
        return None

    # Win probability
    win_p = score / num_games

    # The Elo formula for expected score can be inverted to find the rating difference.
    # E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    # If E_A is our win_p, then:
    # R_A = R_B - 400 * log10(1/win_p - 1)
    if win_p == 1.0:
        # Our model is much stronger, rating is at least 400 points higher
        return opponent_elo + 400
    if win_p == 0.0:
        # Our model is much weaker, rating is at least 400 points lower
        return opponent_elo - 400

    elo_diff = -400 * math.log10(1 / win_p - 1)
    return opponent_elo + elo_diff

def get_stockfish_move(stockfish_engine, board):
    stockfish_engine.set_fen_position(board.fen())
    best_move_uci = stockfish_engine.get_best_move()
    if best_move_uci:
        return chess.Move.from_uci(best_move_uci)
    return None

def get_az_move(az_network, board):
    root_node = mcts.run_mcts(board, az_network, simulations=config.MCTS_SIMULATIONS // 4) # Use fewer sims for eval
    if not root_node.children:
        return None # No legal moves
    best_move_idx = max(root_node.children.items(), key=lambda item: item[1].visit_count)[0]
    best_move_uci = utils.INDEX_TO_MOVE_UCI[best_move_idx]
    return chess.Move.from_uci(best_move_uci)

def play_game(az_network, stockfish_engine, az_plays_white):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        if (board.turn == chess.WHITE and az_plays_white) or \
           (board.turn == chess.BLACK and not az_plays_white):
            move = get_az_move(az_network, board)
        else:
            move = get_stockfish_move(stockfish_engine, board)

        if move is None: # No move was made
            break
        board.push(move)

    result = board.result(claim_draw=True)
    if result == '1-0': return 'win' if az_plays_white else 'loss'
    elif result == '0-1': return 'loss' if az_plays_white else 'win'
    else: return 'draw'

def play_match(az_network, stockfish_skill_level, num_games):
    print(f"\n--- Starting match against Stockfish Skill Level {stockfish_skill_level} ---")
    try:
        stockfish = Stockfish(path=config.STOCKFISH_PATH, parameters={"Skill Level": stockfish_skill_level})
    except Exception as e:
        print(f"ERROR: Could not initialize Stockfish. Please check the path in `config.py`. {e}")
        return None

    scores = {'win': 0, 'loss': 0, 'draw': 0}
    for i in range(num_games):
        az_plays_white = (i % 2 == 0)
        color = "White" if az_plays_white else "Black"
        print(f"  Game {i + 1}/{num_games} (AZ plays as {color})...", end='', flush=True)
        game_result = play_game(az_network, stockfish, az_plays_white)
        scores[game_result] += 1
        print(f" Result: {game_result.upper()}")
    return scores

def main(args):
    utils.setup_gpu()

    # 1. Load the trained AlphaZero model
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at '{args.model_path}'")
        return

    print(f"Loading model from '{args.model_path}'...")
    az_network = network.create_model()
    az_network.load_weights(args.model_path)
    print("Model loaded successfully.")

    # 2. Define the skill levels to test against
    skill_levels_to_test = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    print("\nStarting evaluation...")
    for skill in skill_levels_to_test:
        scores = play_match(az_network, skill, args.num_games)

        if scores:
            # Calculate score from AZ's perspective (win=1, draw=0.5, loss=0)
            az_score = scores['win'] + 0.5 * scores['draw']
            opponent_elo = STOCKFISH_SKILL_TO_ELO[skill]

            estimated_elo = calculate_elo(az_score, opponent_elo, args.num_games)

            print(f"  Match Result: {scores['win']}W - {scores['loss']}L - {scores['draw']}D")
            if estimated_elo:
                print(f"  Estimated Elo: {estimated_elo:.0f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a trained AlphaZero-style chess model against Stockfish at various skill levels."
    )
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to the trained model weights file (.h5), e.g., "checkpoints/latest_model.h5".'
    )
    parser.add_argument(
        '--num-games', type=int, default=10,
        help='Number of games to play against each Stockfish skill level.'
    )

    args = parser.parse_args()
    main(args)
