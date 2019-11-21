
"""A module for homework 2. Version 3."""
# noqa: D413

import abc
import heapq
import itertools
from collections import defaultdict

from hw1 import EightPuzzleState, EightPuzzleNode


def eightPuzzleH1(state, goal_state):
    """
    Return the number of misplaced tiles including blank.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """
    # TODO 1:
    sum = 0
    for y, row in enumerate(state.board):
        for x, element in enumerate(row):
            if goal_state.board[y][x] != element:
                sum += 1
    return sum
    pass


def eightPuzzleH2(state, goal_state):
    """
    Return the total Manhattan distance from goal position of all tiles.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """
    # TODO 2:
    sum = 0
    for y, row in enumerate(state.board):
        for x, element in enumerate(row):
            for rows, goal_row in enumerate(goal_state.board):
                if element in goal_row:
                    path =abs(y - rows) + abs(x - goal_row.index(element))
                    sum += path
    return sum


class Frontier(abc.ABC):
    """An abstract class of a frontier."""

    def __init__(self):
        """Create a frontier."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_empty(self):
        """Return True if empty."""
        raise NotImplementedError()

    @abc.abstractmethod
    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        raise NotImplementedError()


class DFSFrontier(Frontier):
    """An example of how to implement a depth-first frontier (stack)."""

    def __init__(self):
        """Create a frontier."""
        self.dictionary = []

    def is_empty(self):
        """Return True if empty."""
        return len(self.dictionary) == 0

    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        for n in self.dictionary:
            # HERE we assume that state implements __eq__() function.
            # This could be improve further by using a set() datastructure,
            # by implementing __hash__() function.
            if n.state == node.state:
                return None
        self.dictionary.append(node)

    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        return self.dictionary.pop()


class GreedyFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state, goal_state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()

        """
        self.h = h_func
        self.goal = goal_state
        self.dictionary = {}
        self.priority_queue = []
        self.counter = itertools.count()
        # TODO: 3
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.

    def is_empty(self):
        return len(self.dictionary) == 0

    def add(self, node):
        hash_value = node.state
        heuristic_cost = self.h(node.state, self.goal)
        count = next(self.counter)
        if hash_value in self.dictionary:
            self.remove(hash_value)
        entry = [heuristic_cost, count, node]
        self.dictionary[hash_value] = entry
        heapq.heappush(self.priority_queue, entry)

    def next(self):
        while self.priority_queue:
            priority, count, node = heapq.heappop(self.priority_queue)
            if not node.DELETED:
                self.dictionary.pop(node.state)
                return node
            if node is None:
                raise KeyError('pop from an empty priority queue')

    def remove(self, hash):
        entry = self.dictionary.pop(hash)
        entry[2].DELETED = True


class AStarFrontier(Frontier):
    """A frontier for greedy search."""
    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()


        """
        self.h = h_func
        self.goal = goal_state
        self.dictionary = {}
        self.priority_queue = []
        self.counter = itertools.count()
        # TODO: 4
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.

    def is_empty(self):
        return len(self.dictionary) == 0

    def add(self, node):
        hash_value = node.state
        heuristic_cost = self.h(node.state, self.goal)
        path_cost = node.path_cost
        uniform_cost = heuristic_cost + path_cost
        count = next(self.counter)
        entry = [uniform_cost, count, node]
        new_is_good = True
        if hash_value in self.dictionary:
            new_is_good = self.remove(entry)
        if new_is_good:
            self.dictionary[hash_value] = entry
            heapq.heappush(self.priority_queue, entry)

    def next(self):
        while self.priority_queue:
            priority, count, node = heapq.heappop(self.priority_queue)
            if not node.DELETED:
                self.dictionary.pop(node.state)
                return node
            if node is None:
                raise KeyError('pop from an empty priority queue')

    def remove(self, new_entry):
        entry = self.dictionary.get(new_entry[2].state)
        if entry[0] > new_entry[0]:
            entry[2].DELETED = True
            return True
        else:
            self.dictionary[new_entry[2].state] = entry
            return False


def _parity(board):
    """Return parity of a square matrix."""
    inversions = 0
    nums = []
    for row in board:
        for value in row:
            nums.append(value)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] != 0 and nums[j] != 0 and nums[i] > nums[j]:
                inversions += 1
    return inversions % 2


def _is_reachable(board1, board2):
    """Return True if two N-Puzzle state are reachable to each other."""
    return _parity(board1) == _parity(board2)


def graph_search(init_state, goal_state, frontier):
    """
    Search for a plan to solve problem.

    Parameters
    ----------
    init_state : EightPuzzleState
        an initial state
    goal_state : EightPuzzleState
        a goal state
    frontier : Frontier
        an implementation of a frontier which dictates the order of exploreation.

    Returns
    ----------
    plan : List[string] or None
        A list of actions to reach the goal, None if the search fails.
        Your plan should NOT include 'INIT'.
    num_nodes: int
        A number of nodes generated in the search.

    """
    if not _is_reachable(init_state.board, goal_state.board):
        return None, 0
    if init_state.is_goal(goal_state.board):
        return [], 0
    num_nodes = 0
    solution = []
    exploded_state = set()
    # Perform graph search
    node = EightPuzzleNode(init_state, action='INIT')
    frontier.add(node)
    num_nodes+=1
    while not frontier.is_empty():
        current_node = frontier.next()
        current_state = current_node.state
        if current_state.is_goal(goal_state.board):
            break
        exploded_state.add(current_state)
        actions = {'u', 'd', 'l', 'r'}
        for action in actions:
            new_state = current_state.successor(action)
            if new_state is not None:
                if new_state not in exploded_state:
                    num_nodes += 1
                    frontier.add(EightPuzzleNode(new_state, current_node, action))
    node = current_node
    for x, element in enumerate(node.trace()):
        action = element.action
        if action is not 'INIT':
            solution.append(action)
    # Test solution
    # st = init_state
    # print("Test solution")
    # print(st)
    # for x, action in enumerate(solution):
    #     st = st.successor(action)
    #     if st.is_goal():
    #         print('Congratuations!')
    # TODO: 5
    return solution, num_nodes


def test_by_hand(verbose=True):
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    init_state = EightPuzzleState.initializeState()
    while not _is_reachable(goal_state.board, init_state.board):
        init_state = EightPuzzleState.initializeState()

    frontier = GreedyFrontier(eightPuzzleH1, goal_state) # Change this to your own implementation.
    #frontier = AStarFrontier(eightPuzzleH2, goal_state) # Change this to your own implementation.
    if verbose:
        print(init_state)
    plan, num_nodes = graph_search(init_state, goal_state, frontier)
    if verbose:
        print(f'A solution is found after generating {num_nodes} nodes.')
    if verbose:
        for action in plan:
            print(f'- {action}')
    return len(plan), num_nodes


def experiment(n=10000):
    """Run experiments and report number of nodes generated."""
    result = defaultdict(list)
    for __ in range(n):
        d, n = test_by_hand(False)
        result[d].append(n)
    max_d = max(result.keys())
    for i in range(max_d + 1):
        n = result[d]
        if len(n) == 0:
            continue
        print(f'{d}, {len(n)}, {sum(n) / len(n)}')


if __name__ == '__main__':
    __, __ = test_by_hand()
    #experiment()  #  run graph search 10000 times and report result.
