import numpy as np


def setup_training(self):
    """Sets up training"""
    self.number_of_episodes = 100
    self.exploration_rate = self.exploration_rate_initial


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """Called once after each time step except the last. Used to collect training
    data and filling the experience buffer.

    Also, the actual learning takes place here.
    """


def end_of_round(self, last_game_state, last_action, events):
    """Called once per agent (?) after the last step of a round."""
    pass


def training_loop(self):
    for episode in range(self.number_of_episodes):
        self.exploration_rate = self.exploration_rate_end + (
            self.exploration_rate_initial - self.exploration_rate_end
        ) * np.exp(-self.exploration_decay_rate * episode)
