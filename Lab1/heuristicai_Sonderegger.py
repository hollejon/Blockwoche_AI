import numpy as np
import random
import game
import sys
 
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
 
def find_best_move(board):
    """
    Heuristic-based agent to find the best move.
    Tries to maximize score by keeping the largest tile in a corner,
    prioritizing merging tiles and maintaining space.
    """
    best_move = -1
    best_score = -float('inf')
   
    # Try all possible moves and evaluate them
    for move in [UP, DOWN, LEFT, RIGHT]:
        new_board = execute_move(move, board)
       
        # If the move is invalid, continue to next move
        if board_equals(board, new_board):
            continue
       
        # Calculate a score for this move based on our heuristic
        score = evaluate_board(new_board)
       
        # Choose the move with the highest score
        if score > best_score:
            best_score = score
            best_move = move
   
    # If no move improves the board, fall back to a random move
    if best_move == -1:
        best_move = find_best_move_random_agent()
   
    return best_move
 
def find_best_move_random_agent():
    return random.choice([UP, DOWN, LEFT, RIGHT])
 
def execute_move(move, board):
    """
    Move and return the grid without a new random tile.
    It won't affect the state of the game in the browser.
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
        sys.exit("No valid move")
 
def board_equals(board, newboard):
    """
    Check if two boards are equal.
    """
    return (newboard == board).all()
 
def evaluate_board(board):
    """
    Evaluate the board state with a heuristic score.
    The score considers:
    - Corner strategy: favor keeping the largest tile in a corner
    - Free spaces: more empty tiles are better
    - Monotonicity: rows/columns should increase or decrease smoothly
    - Merging potential: prioritize moves that allow merges
    """
    score = 0
   
    # Corner strategy: prioritize having the largest tile in a corner
    max_tile = np.max(board)
    if board[0][0] == max_tile or board[0][3] == max_tile or board[3][0] == max_tile or board[3][3] == max_tile:
        score += max_tile * 10
   
    # Free spaces: prioritize boards with more empty tiles
    empty_spaces = len(np.where(board == 0)[0])
    score += empty_spaces * 25
   
    # Monotonicity: prioritize boards where rows/columns are increasing or decreasing
    score += evaluate_monotonicity(board)
   
    # Merging potential: prioritize moves that bring same value tiles next to each other
    score += evaluate_merge_potential(board)
   
    return score
 
def evaluate_monotonicity(board):
    """
    Reward boards that have increasing or decreasing rows/columns.
    Monotonicity means that values in rows or columns should always increase or decrease.
    The more monotonic, the higher the reward.
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
   
    return score * 100  # weight the score more heavily
 
 
def evaluate_merge_potential(board):
    """
    Reward moves that bring two tiles with the same value together.
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