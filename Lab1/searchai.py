import random
import game
import sys
import numpy as np

# Directions for moves
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Depth limit for the Expectimax search (can be adjusted based on performance)
MAX_DEPTH = 3

def find_best_move(board):
    """
    find the best move for the next turn using Expectimax.
    """
    bestmove = -1
    move_args = [UP, DOWN, LEFT, RIGHT]
    
    # Evaluate all possible moves and their scores using Expectimax
    result = [score_toplevel_move(move, board) for move in move_args]
    
    # Choose the move with the highest score
    bestmove = result.index(max(result))

    for move in move_args:
        print("move: %d score: %.4f" % (move, result[move]))

    return bestmove

def score_toplevel_move(move, board):
    """
    Entry point to score the first move using Expectimax.
    """
    new_board = execute_move(move, board)
    
    if board_equals(board, new_board):
        return 0  # Invalid move, return a score of 0
    
    # Evaluate the move using Expectimax
    return expectimax(new_board, depth=1, is_maximizing_player=False)

def expectimax(board, depth, is_maximizing_player):
    """
    Expectimax algorithm:
    - At maximizing nodes (AI's turn), choose the move with the highest score.
    - At chance nodes (game adds a 2 or 4), calculate the expected score based on the probabilities of adding 2 or 4.
    """
    # Base case: If max depth is reached or the game is over, return heuristic evaluation
    if depth == MAX_DEPTH or np.count_nonzero(board == 0) == 0:
        return evaluate_board(board)

    if is_maximizing_player:
        # Maximizer (AI's turn)
        best_score = -float('inf')
        for move in [UP, DOWN, LEFT, RIGHT]:
            new_board = execute_move(move, board)
            
            # Skip invalid moves
            if board_equals(board, new_board):
                continue
            
            score = expectimax(new_board, depth + 1, False)  # Switch to chance node
            best_score = max(best_score, score)
        
        return best_score
    else:
        # Chance node (random 2 or 4 tile is added)
        empty_tiles = np.argwhere(board == 0)  # Get all empty tile positions
        
        if len(empty_tiles) == 0:
            return evaluate_board(board)
        
        expected_score = 0
        
        # For each empty tile, we expect either a 2 (90% chance) or a 4 (10% chance) to appear
        for tile in empty_tiles:
            i, j = tile
            
            # Add a 2 tile and evaluate
            board_2 = board.copy()
            board_2[i][j] = 2
            score_2 = expectimax(board_2, depth + 1, True)
            
            # Add a 4 tile and evaluate
            board_4 = board.copy()
            board_4[i][j] = 4
            score_4 = expectimax(board_4, depth + 1, True)
            
            # Weighted sum (90% chance for 2, 10% for 4)
            expected_score += 0.9 * score_2 + 0.1 * score_4
        
        # Average the expected score over all empty tiles
        expected_score /= len(empty_tiles)
        
        return expected_score

def execute_move(move, board):
    """
    Executes the move and returns the resulting board state. Doesn't add a new tile.
    """
    if move == UP:
        return game.merge_up(board)
    elif move == DOWN:
        return game.merge_down(board)
    elif move == LEFT:
        return game.merge_left(board)
    elif move == RIGHT:
        return game.merge_right(board)
    else:
        sys.exit("Invalid move")

def board_equals(board, new_board):
    """
    Check if two boards are equal (used to detect invalid moves).
    """
    return (new_board == board).all()

def evaluate_board(board):
    """
    Evaluate the board state using a heuristic score:
    - Corner strategy (priority on keeping the largest tile in a corner)
    - Free spaces (more empty tiles = better)
    - Monotonicity (smooth increase/decrease in rows/columns)
    - Merge potential (bring similar tiles together)
    """
    score = 0
    
    # Corner strategy: prioritize largest tile in a corner
    max_tile = np.max(board)
    if board[0][0] == max_tile or board[0][3] == max_tile or board[3][0] == max_tile or board[3][3] == max_tile:
        score += max_tile * 10
    
    # Free spaces: prioritize having more empty spaces
    empty_spaces = len(np.where(board == 0)[0])
    score += empty_spaces * 25
    
    # Monotonicity: reward rows/columns that are smoothly increasing or decreasing
    score += evaluate_monotonicity(board)
    
    # Merge potential: reward bringing similar tiles together
    score += evaluate_merge_potential(board)
    
    return score

def evaluate_monotonicity(board):
    """
    Reward boards that have increasing or decreasing rows/columns.
    Monotonicity means values should increase or decrease smoothly.
    """
    score = 0

    def calculate_monotonicity(line):
        mono_incr = sum(line[i] <= line[i + 1] for i in range(len(line) - 1))
        mono_decr = sum(line[i] >= line[i + 1] for i in range(len(line) - 1))
        return max(mono_incr, mono_decr)
    
    # Check rows
    for row in board:
        score += calculate_monotonicity(row)
    
    # Check columns
    for col in board.T:
        score += calculate_monotonicity(col)
    
    return score * 100  # Weight the score more heavily

def evaluate_merge_potential(board):
    """
    Reward bringing similar tiles together (merge potential).
    """
    score = 0
    
    # Check rows for merge potential
    for row in board:
        for i in range(len(row) - 1):
            if row[i] == row[i + 1]:
                score += row[i] * 2
    
    # Check columns for merge potential
    for col in board.T:
        for i in range(len(col) - 1):
            if col[i] == col[i + 1]:
                score += col[i] * 2
    
    return score
