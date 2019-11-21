"""A module for homework 1 by Thanapon Noraset."""
import random
import copy


class EightPuzzleState:
    """A class for a state of an 8-puzzle game."""

    def __init__(self, board):
        """Create an 8-puzzle state."""
        self.action_space = {'u', 'd', 'l', 'r'}
        self.board = board
        for i, row in enumerate(self.board):
            for j, v in enumerate(row):
                if v == 0:
                    self.y = i
                    self.x = j

    
    def __repr__(self):
        """Return a string representation of a board."""
        output = []
        for row in self.board:
            row_string = ' | '.join([str(e) for e in row])
            output.append(row_string)
        return ('\n' + '-' * len(row_string) + '\n').join(output)

    def __str__(self):
        """Return a string representation of a board."""
        return self.__repr__()

    @staticmethod
    def initializeState():
        """
        Create an 8-puzzle state with a SHUFFLED tiles.
        
        Returns
        ----------
        EightPuzzleState
            A state that contain an 8-puzzle board with a type of List[List[int]]: 
            a nested list containing integers representing numbers on a board
            e.g., [[0, 1, 2], [3, 4, 5], [6, 7, 8]] where 0 is a blank tile.

        """
        # TODO: 1
        a = list(range(9))
        random.shuffle(a)
        board = [[],[],[]]
        for i in range(9):
            board[i%3].append(a[i]) 
        return EightPuzzleState(board)

    def successor(self, action):
        """
        Move a blank tile in the current state, and return a new state.

        Parameters
        ----------
        action:  string 
            Either 'u', 'd', 'l', or 'r'.

        Returns
        ----------
        EightPuzzleState or None
            A resulting 8-puzzle state after performing `action`.
            If the action is not possible, this method will return None.

        Raises
        ----------
        ValueError
            if the `action` is not in the action space
        
        """    
        if action not in self.action_space:
            raise ValueError(f'`action`: {action} is not valid.')
        # TODO: 2
        # YOU NEED TO COPY A BOARD BEFORE MODIFYING IT
        new_board = copy.deepcopy(self.board)

        if action == 'u':
            if self.y == 0:
                return None
            new_board[self.y][self.x] = self.board[self.y-1][self.x]
            new_board[self.y-1][self.x] = 0
        elif action == 'd':
            if self.y == 2:
                return None
            new_board[self.y][self.x] = self.board[self.y+1][self.x]
            new_board[self.y+1][self.x] = 0
        elif action == 'l':
            if self.x == 0:
                return None
            new_board[self.y][self.x] = self.board[self.y][self.x-1]
            new_board[self.y][self.x-1] = 0
        elif action == 'r':
            if self.x == 2:
                return None
            new_board[self.y][self.x] = self.board[self.y][self.x+1]
            new_board[self.y][self.x+1] = 0
        return EightPuzzleState(new_board)

    def is_goal(self, goal_board=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]):
        """
        Return True if the current state is a goal state.
        
        Parameters
        ----------
        goal_board (optional)
            The desired state of 8-puzzle.

        Returns
        ----------
        Boolean
            True if the current state is a goal.
        
        """
        # TODO: 3
        return self.board == goal_board

    def __eq__(self, state):
        """Return True if the `state` has the same board."""
        if not isinstance(state, EightPuzzleState):
            return False
        return self.board == state.board

    def __hash__(self):
        """Return a hash value of the board."""
        i = 1
        h = 0
        for row in self.board:
            for v in row:
                h += v * i
                i *= 10
        return h


class EightPuzzleNode:
    """A class for a node in a search tree of 8-puzzle state."""
    
    def __init__(
            self, state, parent=None, action=None, cost=1):
        """Create a node with a state."""
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.DELETED = False
        self.newEntry = False
        self.duplicate = None
        if parent is not None:
            self.path_cost = parent.path_cost + self.cost
        else:
            self.path_cost = 0

    def trace(self):
        """
        Return a path from the root to this node.

        Returns
        ----------
        List[EightPuzzleNode]
            A list of nodes stating from the root node to the current node.

        """
        # TODO: 4
        cur_node = self
        a = [cur_node]
        while cur_node.parent is not None:
            cur_node = cur_node.parent
            a.append(cur_node)
        return list(reversed(a))


def test_by_hand():
    """Run a CLI 8-puzzle game."""
    state = EightPuzzleState.initializeState()
    root_node = EightPuzzleNode(state, action='INIT')
    cur_node = root_node
    print(state)
    action = input('Please enter the next move (q to quit): ')
    while action != 'q':
        new_state = cur_node.state.successor(action)
        cur_node = EightPuzzleNode(new_state, cur_node, action)
        print(new_state)
        if new_state.is_goal():
            print('Congratuations!')
            break
        action = input('Please enter the next move (q to quit): ')
    
    print('Your actions are: ')
    for node in cur_node.trace():
        print(f'  - {node.action}')
    print(f'The total path cost is {cur_node.path_cost}')

if __name__ == '__main__':
    test_by_hand()