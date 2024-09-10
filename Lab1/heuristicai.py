import random
import sys
import numpy as np
import game

# Game move directions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Global weights, dynamically updated during the game
CORNER_WEIGHT = 150
FREE_SPACE_WEIGHT = 120
MONOTONICITY_WEIGHT = 100
MERGE_POTENTIAL_WEIGHT = 50
ISOLATION_PENALTY_WEIGHT = 80

LOOKAHEAD_DEPTH = 1  # Default depth for lookahead

def find_best_move(board):
    best_move = -1
    best_score = -float('inf')
    
    empty_tiles = len(np.where(board == 0)[0])
    update_weights(board)  # Adjust weights dynamically based on game stage
    global LOOKAHEAD_DEPTH
    LOOKAHEAD_DEPTH = adjust_lookahead_depth(empty_tiles)  # Adjust lookahead depth

    # Try all moves and evaluate their outcomes
    for move in [UP, DOWN, LEFT, RIGHT]:
        new_board = execute_move(move, board)
        
        if board_equals(board, new_board):
            continue

        score = evaluate_board(new_board, depth=LOOKAHEAD_DEPTH)
        
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

def evaluate_board(board, depth=1):
    if depth == 0:
        return evaluate_heuristics(board)
    
    max_score = -float('inf')
    
    for move in [UP, DOWN, LEFT, RIGHT]:
        new_board = execute_move(move, board)
        if board_equals(board, new_board):
            continue
        
        score = evaluate_heuristics(new_board)
        score += evaluate_board(new_board, depth - 1)  # Recursive lookahead

        if score > max_score:
            max_score = score

    return max_score

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
    if board[3][0] == max_tile or board[3][3] == max_tile:
        return max_tile
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
    penalty = 0
    max_tile = np.max(board)
    
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
    max_tile = np.max(board)
    gradient_score = 0

    # Simple check: is the gradient preserved toward a corner?
    for row in range(3):
        for col in range(3):
            if board[row][col] < board[row + 1][col] or board[row][col] < board[row][col + 1]:
                gradient_score += board[row][col]

    return gradient_score

def update_weights(board):
    global CORNER_WEIGHT, FREE_SPACE_WEIGHT, MONOTONICITY_WEIGHT, MERGE_POTENTIAL_WEIGHT, ISOLATION_PENALTY_WEIGHT
    max_tile = np.max(board)
    if max_tile < 256:  # Early Game
        CORNER_WEIGHT = 80
        FREE_SPACE_WEIGHT = 150
        MONOTONICITY_WEIGHT = 60
        MERGE_POTENTIAL_WEIGHT = 40
        ISOLATION_PENALTY_WEIGHT = 10
    elif 256 >= max_tile <= 1024:  # Mid Game
        CORNER_WEIGHT = 120
        FREE_SPACE_WEIGHT = 190
        MONOTONICITY_WEIGHT = 120
        MERGE_POTENTIAL_WEIGHT = 50
        ISOLATION_PENALTY_WEIGHT = 20
    else:  # Late Game
        CORNER_WEIGHT = 160
        FREE_SPACE_WEIGHT = 210
        MONOTONICITY_WEIGHT = 175
        MERGE_POTENTIAL_WEIGHT = 70
        ISOLATION_PENALTY_WEIGHT = 70

def adjust_lookahead_depth(empty_tiles):
    if empty_tiles > 8:
        return 1  # Early game, shallow lookahead
    elif 4 <= empty_tiles <= 8:
        return 2  # Mid game, deeper lookahead
    else:
        return 3  # Late game, deepest lookahead to avoid traps