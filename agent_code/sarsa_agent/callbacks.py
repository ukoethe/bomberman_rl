import os
import pickle
import random
import numpy as np
from .train import QValue
from .features_actual import BombermanFeatures
from sklearn.tree import DecisionTreeRegressor

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUM_FEATURES = 19

def setup(agent):
    agent.feature_extractor = BombermanFeatures()

    if agent.train or not os.path.isfile("my-saved-model.pt"):
        agent.logger.info("Setting up model from scratch.")
        agent.decision_trees = [DecisionTreeRegressor() for _ in range(len(ACTIONS))]
    else:
        agent.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            agent.decision_trees = pickle.load(file)

def act(agent, game_state: dict) -> str:
    if agent.train and (random.random() < agent.EPSILON or game_state['round'] < 50):
        agent.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    Q_values = [agent.decision_trees[action_idx].predict([state_to_features(agent, game_state)])[0] for action_idx in range(len(ACTIONS))]
    action_idx = np.argmax(Q_values)

    return ACTIONS[action_idx]

def state_to_features(agent, game_state: dict) -> np.array:
    if game_state is None:
        return None

    return agent.feature_extractor.state_to_features(game_state)
