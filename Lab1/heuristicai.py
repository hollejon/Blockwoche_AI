import random
import sys
import numpy as np
import game

# Game move directions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Global weights, dynamically updated during the game
CORNER_WEIGHT = 100
FREE_SPACE_WEIGHT = 100
MONOTONICITY_WEIGHT = 100
MERGE_POTENTIAL_WEIGHT = 100
ISOLATION_PENALTY_WEIGHT = 100

LOOKAHEAD_DEPTH = 1  # Default depth for lookahead

def find_best_move(board):
    best_move = -1
    best_score = -float('inf')
    
    # Find the position of the max tile
    max_tile = np.max(board)
    max_tile_position = np.where(board == max_tile)
    
    if max_tile_position[0].size > 0:
        row, col = max_tile_position[0][0], max_tile_position[1][0]
    
    empty_tiles = len(np.where(board == 0)[0])
    update_weights(empty_tiles, board)


    # Try all moves and evaluate their outcomes
    for move in [UP, DOWN, LEFT, RIGHT]:
        new_board = execute_move(move, board)

        if board_equals(board, new_board):
            continue

        # Check if the max tile will stay in a corner after the move
        if (move == UP and (row == 0 or col == 0)) or \
           (move == DOWN and (row == 3 or col == 3)) or \
           (move == LEFT and (col == 0 or row == 0)) or \
           (move == RIGHT and (col == 3 or row == 3)):
            score = evaluate_heuristics(board) + 200  # Bonus for maintaining corner position
        else:
            score = evaluate_heuristics(board)

        if score > best_score:
            best_score = score
            best_move = move

    if best_move == -1:
        best_move = find_best_move_random_agent(board)

    return best_move

def find_best_move_random_agent(board):
    valid_moves = []
    for move in [UP, DOWN, LEFT, RIGHT]:
        new_board = execute_move(move, board)
        if not board_equals(board, new_board):
            valid_moves.append(move)
    
    return random.choice(valid_moves) if valid_moves else random.choice([UP, DOWN, LEFT, RIGHT])

def execute_move(move, board):
    # Execute the move logic (assume a valid implementation of game.move functions)
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

def board_equals(board, new_board):
    return (new_board == board).all()


def evaluate_heuristics(board):
    score = 0

    # Corner strategy
    score += evaluate_corner_strategy(board) * CORNER_WEIGHT

    # Free spaces
    score += evaluate_free_spaces(board) * FREE_SPACE_WEIGHT

    # Monotonicity
    score += evaluate_monotonicity(board) * MONOTONICITY_WEIGHT

    # Merge potential
    score += evaluate_merge_potential(board) * MERGE_POTENTIAL_WEIGHT

    # Isolation penalty
    score -= evaluate_isolation_penalty(board) * ISOLATION_PENALTY_WEIGHT

    # Tile distribution awareness
    score += evaluate_tile_distribution(board) * 50  # Adjust the weight based on testing

    return score

def evaluate_corner_strategy(board):
    max_tile = np.max(board)
    # Check if the max tile is in the bottom-left corner (3, 0)
    if board[3][0] == max_tile or board[0][0] == max_tile :
        return 100  # Strongly reward if the max tile is in the corner
    # Check if the max tile is in the bottom-right corner (3, 3)
    elif board[3][3] == max_tile or board[0][3] == max_tile :
        return 100  # Strongly reward if the max tile is in the corner
    return 0

def evaluate_free_spaces(board):
    return len(np.where(board == 0)[0])

def evaluate_monotonicity(board):
    score = 0

    def calculate_monotonicity(line):
        mono_incr = sum(line[i] <= line[i + 1] for i in range(len(line) - 1))
        mono_decr = sum(line[i] >= line[i + 1] for i in range(len(line) - 1))
        return max(mono_incr, mono_decr)

    for row in board:
        score += calculate_monotonicity(row)
    
    for col in board.T:
        score += calculate_monotonicity(col)

    return score

def evaluate_merge_potential(board):
    score = 0
    
    for row in board:
        for i in range(len(row) - 1):
            if row[i] == row[i + 1]:
                score += row[i]
    
    for col in board.T:
        for i in range(len(col) - 1):
            if col[i] == col[i + 1]:
                score += col[i]

    return score

def evaluate_isolation_penalty(board):
    
    max_tile = np.max(board)
    penalty = 0

    
    
    max_tile_position = np.where(board == max_tile)
    if len(max_tile_position[0]) > 0:
        row, col = max_tile_position[0][0], max_tile_position[1][0]
        adjacent_tiles = []
        if row > 0:
            adjacent_tiles.append(board[row - 1][col])
        if row < 3:
            adjacent_tiles.append(board[row + 1][col])
        if col > 0:
            adjacent_tiles.append(board[row][col - 1])
        if col < 3:
            adjacent_tiles.append(board[row][col + 1])
        
        for tile in adjacent_tiles:
            if tile == 0 or tile < max_tile / 4:
                penalty += 1

    return penalty

def evaluate_tile_distribution(board):
    """
    This function evaluates how well-distributed the tiles are in a gradient from one corner.
    Ideally, the largest tiles should be in one corner and the values should decrease outward.
    """
    gradient_score = 0
    max_tile = np.max(board)
    
    # Encourage lower values to be further from the corner
    for row in range(4):
        for col in range(4):
            if board[row][col] == 0:
                continue
            distance_to_corner = (3 - row) + (3 - col)  # Distance from bottom-left corner (3, 0)
            if distance_to_corner > 2 and board[row][col] < max_tile / 2:
                gradient_score += (max_tile / 2 - board[row][col])  # Penalize low tiles near corners
            else:
                gradient_score += board[row][col]  # Reward higher tiles

    return gradient_score

def update_weights(empty_tiles, board):
    global CORNER_WEIGHT, FREE_SPACE_WEIGHT, MONOTONICITY_WEIGHT, MERGE_POTENTIAL_WEIGHT, ISOLATION_PENALTY_WEIGHT

    max_tile = np.max(board)

    if max_tile < 256:  # Early Game
        CORNER_WEIGHT = 300
        FREE_SPACE_WEIGHT = 150
        MONOTONICITY_WEIGHT = 50
        MERGE_POTENTIAL_WEIGHT = 1
        ISOLATION_PENALTY_WEIGHT = 1
    elif 256 >= max_tile <= 1024:  # Mid Game
        CORNER_WEIGHT = 300
        FREE_SPACE_WEIGHT = 150
        MONOTONICITY_WEIGHT = 50
        MERGE_POTENTIAL_WEIGHT = 1
        ISOLATION_PENALTY_WEIGHT = 1
    else:  # Late Game
        CORNER_WEIGHT = 300
        FREE_SPACE_WEIGHT = 150
        MONOTONICITY_WEIGHT = 50
        MERGE_POTENTIAL_WEIGHT = 1
        ISOLATION_PENALTY_WEIGHT = 1