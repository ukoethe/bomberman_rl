import os
import dill as pickle
from .train import train_act
from .utils import state_to_features, ACTIONS


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

    # check if we are in training mode and if the model already exists
    if self.train or not os.path.isfile("model.pt"):

        self.logger.info("Training model.")

        # Deactivate if u want to train a completely new agent
        if os.path.isfile("model.pt"):
            self.continue_train = True
        else:
            self.continue_train = False

        self.action_space_size = len(ACTIONS)

    else:
        self.logger.info("Loading model from saved state.")
        with open("model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    features = state_to_features(game_state)

    if self.train:
        action = train_act(self, game_state)
        return action

    # self.logger.debug("Querying model for action")
    action = self.model.choose_action(features)
    self.logger.debug(f"Model returned action: {action}")

    return action

