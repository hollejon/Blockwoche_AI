import random
import game
import sys

# Author:				chrn (original by nneonneo)
# Date:				11.11.2016
# Description:			The logic of the AI to beat the game.

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

def find_best_move(board):
    bestmove = -1    
	
	# TODO:
	# Build a heuristic agent on your own that is much better than the random agent.
	# Your own agent don't have to beat the game.
    # bestmove = find_best_move_random_agent()
    bestmove = find_best_move_ai_agent(board)
    return bestmove

def find_best_move_random_agent():
    return random.choice([UP,DOWN,LEFT,RIGHT])
    
def find_best_move_ai_agent(board):

    # returns an array of all the empty tiles
    def locate_empty_tiles (board):
        empty_tiles = []
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    empty_tiles.append((i,j))
        return empty_tiles
    
    # returns the number of empty tiles -> W1
    def number_empty_tiles (board):
        return len(locate_empty_tiles(board))
    
    # locates the adjacent tiles when given a position  
    def locate_adjacent_tiles (position):
        adjacent_tiles = []
        if position[0] > 0:
            adjacent_tiles.append((position[0]-1, position[1]))
        if position[0] < 3:
            adjacent_tiles.append((position[0]+1, position[1]))
        if position[1] > 0:
            adjacent_tiles.append((position[0], position[1]-1))
        if position[1] < 3:
            adjacent_tiles.append((position[0], position[1]+1))
        return adjacent_tiles
    
    #def find_move_with_most_merged_tiles(board):


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