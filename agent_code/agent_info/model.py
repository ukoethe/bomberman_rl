import numpy as np
from .utils import state_to_features
from collections import defaultdict


class Q_Table:
    def __init__(self, game, actions) -> None:

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.01
        self.min_exploration = 0.6
        self.exploration_decay = 0.0001

        self.game = game

        self.actions = np.array(actions)

        self.q_table = defaultdict(
            lambda: np.zeros([game.action_space_size])
        )  # We should start small and build as goes for faster look ups and less memory usage

    def choose_action(self, features):

        action_index = np.argmax(self.q_table[features])
        return self.actions[action_index]

    def update_q(self, old_game_state, new_game_state, self_action, rewards):

        actionIndex = np.where(self.actions == self_action)[0][0]
        old_ft = state_to_features(old_game_state)
        new_ft = state_to_features(new_game_state)

        if type(old_ft) is not tuple:
            self.game.logger.debug(f"{old_ft} was not a tuple")
            return None

        if type(new_ft) is not tuple:
            self.game.logger.debug(f"{new_ft} was not a tuple")
            return None

        # We do not have to check if either of those exist
        # If they dont default dict creates an np.zero array for us with that key
        exspected_reward = self.q_table[old_ft][actionIndex]
        max_next_reward = np.max(self.q_table[new_ft])

        # The actual Q update step based on temporal difference
        updated_q = (1 - self.alpha) * exspected_reward + self.alpha * (
            rewards + self.gamma * max_next_reward
        )
        self.q_table[old_ft][actionIndex] = updated_q

        if self.epsilon < 1 - self.min_exploration:
            self.epsilon += self.exploration_decay

    def update_terminal(self, old_game_state, self_action, rewards):

        action_index = self.actions[self_action]
        old_ft = state_to_features(old_game_state)

        if type(old_ft) is not tuple:
            self.game.logger.debug(f"{old_ft} was not a tuple")
            return None

        # We do not have to check if either of those exist
        # If they dont default dict creates an np.zero array for us with that key
        exspected_reward = self.q_table[old_ft][action_index]

        # The actual Q update step based on temporal difference
        updated_q = (1 - self.alpha) * exspected_reward + self.alpha * rewards

        self.q_table[old_ft][action_index] = updated_q
