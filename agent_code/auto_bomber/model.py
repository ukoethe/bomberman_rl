import pickle
from pathlib import Path

import numpy as np

import agent_code.auto_bomber.auto_bomber_config as config
from agent_code.auto_bomber.transitions import Transitions
from math import exp
import agent_code.auto_bomber.auto_bomber_config as config

class LinearAutoBomberModel:
    def __init__(self, feature_extractor):
        self.weights = None
        self.feature_extractor = feature_extractor

        path = Path(config.MODEL_PATH)
        if path.is_file():
            with path.open(mode="rb") as file:
                self.weights = pickle.load(file)

    def store(self):
        path = Path(config.MODEL_PATH)
        with path.open(mode="wb") as file:
            pickle.dump(self.weights, file)

    def select_best_action(self, game_state: dict, agent_self, softmax=False):
        features_x = self.feature_extractor(game_state)
        self.init_if_needed(features_x, agent_self)
        q_action_values = np.sum(self.weights.transpose() * features_x[:, np.newaxis], axis=0)

        if softmax:
            sort_actions = q_action_values.argsort()
            p = np.exp(sort_actions / config.TEMP) / np.sum(np.exp(sort_actions / config.TEMP))
            choice = np.random.choice(sort_actions, p=p)
        else:
            top_3_actions = q_action_values.argsort()[-3:][::-1]
            choice = np.random.choice(top_3_actions, p=[0.9, 0.05, 0.05])
        return config.ACTIONS[choice]

    def fit_model_with_transition_batch(self, transitions: Transitions):
        for action_id, action in enumerate(config.ACTIONS):
            numpy_transitions = transitions.to_numpy_transitions()
            x_all_t, y_all_t = numpy_transitions.get_features_and_value_estimates(action)

            if x_all_t.size != 0:
                q_estimations = np.sum(x_all_t * self.weights[action_id], axis=0)
                residuals = (y_all_t - q_estimations[:, np.newaxis])
                q_grad = np.sum(x_all_t.transpose() * residuals, axis=1)

                weight_updates = config.LEARNING_RATE / y_all_t.shape[0] * q_grad
                self.weights[action_id] += weight_updates

    def init_if_needed(self, features_x, agent_self):
        if self.weights is None:
            agent_self.logger.info("Model is empty init with random weights.")
            # new_weights = np.random.rand(len(config.ACTIONS), len(features_x))
            # self.weights = new_weights / new_weights.sum(axis=0)  # all weights are 0 < weight < 1
            self.weights = np.ones((len(config.ACTIONS), len(features_x)))
