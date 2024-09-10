import random
import game
import sys
import numpy as np

# Author:				chrn (original by nneonneo)
# Date:				11.11.2016
# Description:			The logic of the AI to beat the game.

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MAX_DEPTH = 3  # Depth limit for minimax search

'''
def find_best_move(board):
    bestmove = -1    
	
	# TODO:
	# Build a heuristic agent on your own that is much better than the random agent.
	# Your own agent don't have to beat the game.
    bestmove = find_best_move_random_agent()
    return bestmove
'''


# The evaluation function considering the monotonicity, smoothness, and number of free tiles.
def evaluate_board(board):
    return monotonicity(board) + smoothness(board) + free_tiles(board)

def monotonicity(board):
    """
    Heuristic to check whether the board values are arranged in a monotonically increasing 
    or decreasing order along both left/right and up/down directions.
    """
    total_score = 0
    # We want to encourage values to be high in one corner
    # We check the monotonicity across rows and columns
    for row in board:
        for i in range(len(row) - 1):
            if row[i] >= row[i+1]:
                total_score += row[i+1] - row[i]
            else:
                total_score -= row[i] - row[i+1]
                
    for col in board.T:
        for i in range(len(col) - 1):
            if col[i] >= col[i+1]:
                total_score += col[i+1] - col[i]
            else:
                total_score -= col[i] - col[i+1]
    return total_score

def smoothness(board):
    """
    Heuristic that rewards boards where adjacent tiles have smaller differences, 
    so they are easier to merge.
    """
    total_score = 0
    for i in range(4):
        for j in range(3):
            total_score -= abs(board[i][j] - board[i][j + 1])  # Horizontal smoothness
            total_score -= abs(board[j][i] - board[j + 1][i])  # Vertical smoothness
    return total_score

def free_tiles(board):
    """
    Heuristic to reward states with more empty (0) tiles.
    """
    return len(np.where(board == 0)[0]) * 100  # Reward heavily for free tiles

# Minimax search with alpha-beta pruning
def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or game_over(board):
        return evaluate_board(board)

    if maximizing_player:
        max_eval = -float('inf')
        for move in [UP, DOWN, LEFT, RIGHT]:
            new_board = execute_move(move, board)
            if not board_equals(board, new_board):  # Only evaluate new moves
                eval = minimax(new_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = float('inf')
        # Random tile placement (AI's turn to add a 2 or 4 in an empty cell)
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    for tile_value in [2, 4]:
                        new_board = board.copy()
                        new_board[i][j] = tile_value
                        eval = minimax(new_board, depth - 1, alpha, beta, True)
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
        return min_eval

def find_best_move(board):
    """
    Determines the best move to make using minimax search with alpha-beta pruning.
    """
    best_move = -1
    best_score = -float('inf')
    for move in [UP, DOWN, LEFT, RIGHT]:
        new_board = execute_move(move, board)
        if not board_equals(board, new_board):  # Only evaluate valid moves
            move_score = minimax(new_board, MAX_DEPTH, -float('inf'), float('inf'), False)
            if move_score > best_score:
                best_score = move_score
                best_move = move
    return best_move

# Functions provided for game mechanics
def execute_move(move, board):
    """
    Executes a move and returns the resulting board state.
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

def board_equals(board, newboard):
    """
    Checks if two boards are identical.
    """
    return (newboard == board).all()

def game_over(board):
    """
    Checks if no valid moves are available, meaning the game is over.
    """
    for move in [UP, DOWN, LEFT, RIGHT]:
        if not board_equals(board, execute_move(move, board)):
            return False
    return True


def find_best_move_random_agent():
    return random.choice([UP,DOWN,LEFT,RIGHT])
    
def execute_move(move, board):
    """
    move and return the grid without a new random tile 
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
    Check if two boards are equal
    """
    return  (newboard == board).all()  