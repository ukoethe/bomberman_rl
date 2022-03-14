from sre_parse import State
import numpy as np
from collections import defaultdict

def MonteCarloTree():

    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._no_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = self.untried_actions()
    
    def untried_actions(self):
        return self.state.get_legal_actions()

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]

        return wins - loses

    def n(self):
        return self._no_visits

    def expand(self):

        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTree(next_state, parent = self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):

        current_rollout_state = self.state

        while not current_rollout_state.is_game_over():

            possible_moves = current_rollout_state.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)

        return current_rollout_state.game_result()

    def backpropagte(self, result):
        pass

    def is_fully_expanded(self):
        pass

    def ...()