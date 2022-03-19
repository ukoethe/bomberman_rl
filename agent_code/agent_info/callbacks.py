from collections import defaultdict
import os
import pickle
import random
from typing import Dict, List, Tuple
from unicodedata import category
from .train import setup_training, train_act
from .utils import state_to_features
from .model import Q_Table
import numpy as np
from random import shuffle


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


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
    if self.train or not os.path.isfile("model.pt"):

        self.logger.info("Training model.")

        # Deactivate if u want to train a completly new agent
        if os.path.isfile("model.pt"):
            self.continue_train = True
        else:
            self.continue_train = False

        self.action_space_size = len(ACTIONS)
        setup_training(self)

    else:
        self.logger.info("Loading model from saved state.")
        self.model = pickle.load("model.pt")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train:
        action = train_act(self, game_state)
        return action

    self.logger.debug("Querying model for action")

    action_index = self.model.chose_action(game_state)
    model_action = self.model.actions[action_index]
    self.logger.debug("Model returnd action: ", model_action)

    self.logger.debug("Querying model for action.")
    return model_action

