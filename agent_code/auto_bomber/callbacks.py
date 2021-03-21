import random

import numpy as np

from agent_code.auto_bomber.feature_engineering import state_to_features
from agent_code.auto_bomber.model import LinearAutoBomberModel


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.model = LinearAutoBomberModel(self.train, feature_extractor=lambda x: state_to_features(x))


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    hyper_parameters = self.model.hyper_parameters
    # todo right now epsilon-greedy - change to softmax to avoid local maxima
    if self.train and random.random() < hyper_parameters["epsilon"]:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(hyper_parameters["actions"], p=[.2, .2, .2, .2, .1, .1])
    else:
        self.logger.debug("Querying model for action.")
        return self.model.select_best_action(game_state, self)
