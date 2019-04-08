"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return depth_score(game,player)

def closeToCenterAndMaxMoves_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Return minus infinity if player loses
    if game.is_loser(player):
        return float("-inf")
    # Return infinity if player wins
    if game.is_winner(player):
        return float("inf")

    # Gets the current position of the player
    position = game.get_player_location(player)
    # Calculates distance to the center

    # Number of allowed moves for the current player
    own_moves = len(game.get_legal_moves(player))

    # Return the value associated to the current state of the board: distance minimization combined with maximization of allowed moves
    return -float(math.sqrt((position[1] + 1 - game.width / 2.0) ** 2 + (position[0] + 1 - game.height / 2.0) ** 2)) + (own_moves)

def closeToCenter_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #Return minus infinity if player loses
    if game.is_loser(player):
        return float("-inf")
    #Return infinity if player wins
    if game.is_winner(player):
        return float("inf")

    # Gets the current position of the player
    position = game.get_player_location(player)
    # Calculates distance to the center
    return -float(math.sqrt((position[1] + 1 - game.width / 2.0) ** 2 + (position[0] + 1 - game.height / 2.0) ** 2))


def farFromEdges_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Return minus infinity if player loses
    if game.is_loser(player):
        return float("-inf")
    # Return infinity if player wins
    if game.is_winner(player):
        return float("inf")

    # Gets the current position of the player
    position = game.get_player_location(player)
    # Calculates distance to the center
    return float(min([position[1], game.width - (position[1] + 1), position[0], game.height - (position[0] + 1)]))

def depth_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Return minus infinity if player loses
    if game.is_loser(player):
        return float("-inf")
    # Return infinity if player wins
    if game.is_winner(player):
        return float("inf")

    #Sample code recommended by Udacity reviewer
    visited = set()
    depth = 4
    path = set(game.get_legal_moves(player))
    score = 0
    for _ in range(depth):
        if len(path) == 0:
            break
        next_path = set()
        for m in path:
            next_path = next_path.union(set(game.__get_moves__(m)))

        visited = visited.union(path)
        path = next_path - visited
        score += 1
    return float(score)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn

        if method == 'minimax':
            self.method = self.minimax
        elif method == 'alphabeta':
            self.method = self.alphabeta

        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            #If self.iterative true perform iterative deepening search until timeout
            if self.iterative:
                depth = 1
                while True:
                    score = self.method(game, depth)
                    depth += 1
            #Else perform fixed depth search
            else:
                score = self.method(game, self.search_depth)
            pass

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass
        # Return the best move from the last completed search iteration or (-1,-1) if no move was found in the time limit
        try:
            return score[1]
        except NameError:
            return (-1, -1)



    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #Timeout when time_left is less than the threshold
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #Get a list with the allowed position in current game state
        allowed_positions = game.get_legal_moves(game.active_player)

        #Return empty list when the depth to search for is 0
        if depth == 0:
            return []

        #When no moves are available return -infinity if maximizing_player is true (he loses) or infinity if false (he wins)
        if len(allowed_positions) == 0:
            if maximizing_player:
                return (float("-inf"), (-1,-1))
            else:
                return (float("inf"), (-1,-1))

        #This list stores all the possible scores for a given node in the tuple (score, node)
        scores = []
        #Go through every allowed_position and use the evaluation function/minimax to find the best possible next move
        for node in allowed_positions:

            #Advance to new state forecasting move to 'node'
            new_game = game.forecast_move(node)

            #If depth to search is 1, directly get the scores of the children nodes
            if depth == 1:
                if maximizing_player:
                    scores += [(self.score(new_game, game.active_player), node)]
                else:
                    scores += [(self.score(new_game, game.get_opponent(game.active_player)),node)]
            else:
                #If depth is not 1, go one level below in the three and apply minimax
                scores += [(self.minimax(new_game, depth-1, not maximizing_player)[0],node)]
        #Sort scores from children nodes from lowest to highest
        scores.sort(key=lambda tup: tup[0])
        if maximizing_player:
            #If maximing player return highest score
            return scores[-1]
        else:
            #If minimizing player return lowest score
            return scores[0]



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        # Timeout when time_left is less than the threshold
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get a list with the allowed position in current game state
        allowed_positions = game.get_legal_moves(game.active_player)

        # Return empty list when the depth to search for is 0
        if depth == 0:
            return []

        # When no moves are available return -infinity if maximizing_player is true (he loses) or infinity if false (he wins)
        if len(allowed_positions) == 0:
            if maximizing_player:
                return (float("-inf"), (-1, -1))
            else:
                return (float("inf"), (-1, -1))

        #Default final node when there is no solution
        nodeF = (-1,-1)

        # Go through every allowed_position and use the evaluation function/AB pruning to find the best possible next move
        for node in allowed_positions:
            # Advance to new state forecasting move to 'node'
            new_game = game.forecast_move(node)

            if maximizing_player:
                # If depth to search is 1, directly get the scores of the children nodes
                if depth == 1:
                    scoreN = self.score(new_game, game.active_player)
                # If depth is not 1, go one level below in the three and apply AB pruning
                else:
                    scoreN = self.alphabeta(new_game, depth=depth - 1,alpha=alpha, beta=beta,maximizing_player=not maximizing_player)[0]
                #Prune when current score is greater or equal to beta for a maximizing parent node
                if scoreN >= beta:
                    return (scoreN, node)
                else:
                    #Change alpha value if score is greater than current upper limit
                    if (scoreN > alpha):
                        alpha = scoreN
                        nodeF = node
            else:
                # If depth to search is 1, directly get the scores of the children nodes
                if depth == 1:
                    scoreN = self.score(new_game, game.get_opponent(game.active_player))
                else:
                    scoreN = self.alphabeta(new_game, depth=depth - 1, alpha=alpha, beta=beta, maximizing_player=not maximizing_player)[0]
                # Prune when current score is less or equal to alpha for a minimizing parent node
                if scoreN <= alpha:
                    return (scoreN,node)
                else:
                    #change beta value if score is less than current lower limit
                    if scoreN < beta:
                        beta = scoreN
                        nodeF = node
        # Sort scores from children nodes from lowest to highest
        if maximizing_player:
            # If maximing player return highest score
            return (alpha, nodeF)
        else:
            # If minimizing player return lowest score
            return (beta, nodeF)
