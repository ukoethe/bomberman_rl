import os
import pickle
import numpy as np
import agent_code.auto_bomber.auto_bomber_config as config


class LinearAutoBomberModel:
    def __init__(self, feature_extractor):
        self.weights = None
        self.feature_extractor = feature_extractor

        if os.path.isfile(config.MODEL_PATH):
            with open(config.MODEL_PATH, "rb") as file:
                self.weights = pickle.load(file)

    def store(self):
        with open(config.MODEL_PATH, "wb") as file:
            pickle.dump(self.weights, file)

    def select_best_action(self, game_state: dict, agent_self):
        features_x = self.feature_extractor(game_state)
        self.init_if_needed(features_x, agent_self)
        q_action_values = np.sum(self.weights.transpose() * features_x[:, np.newaxis], axis=0)

        top_3_actions = q_action_values.argsort()[-3:][::-1]
        # lets keep a little bit randomness here
        choice = np.random.choice(top_3_actions, p=[0.9, 0.05, 0.05])
        return config.ACTIONS[choice]

    def init_if_needed(self, features_x, agent_self):
        if self.weights is None:
            agent_self.logger.info("Model is empty init with random weights.")
            new_weights = np.random.rand(len(config.ACTIONS), len(features_x))
            self.weights = new_weights / new_weights.sum(axis=0)  # all weights are 0 < weight < 1
