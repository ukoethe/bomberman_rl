import numpy as np
from .utils import state_to_features
from collections import defaultdict

import random
import numpy as np
from collections import deque


from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

# from sklearn.classifier_selection import train_test_split

GAMMA = 0.95
LEARNING_RATE = 0.001

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.2
EXPLORATION_DECAY = 0.96


class DQNSolver:
    def __init__(self, game, actions):
        self.exploration_rate = EXPLORATION_MAX
        self.game = game
        self.action_space = len(actions)
        self.actions = actions

        self.classifier = MultiOutputRegressor(
            LGBMRegressor(n_estimators=100, n_jobs=-1)
        )
        # self.classifier = KNeighborsRegressor(n_jobs=-1)
        # self.classifier = MultiOutputRegressor(SVR(), n_jobs=8)
        self.isFit = False

    def experience_replay(self, transitions, batch_size=30):
        if len(transitions) < batch_size:
            return
        batch = random.sample(transitions, int(len(transitions) / 1))
        X = []
        state_size = 31
        targets = []
        for state, action, state_next, reward in batch:
            q_update = reward

            if self.isFit:
                q_update = reward + GAMMA * np.amax(
                    self.classifier.predict(state_next)[0]
                )
                q_values = self.classifier.predict(state)
            else:
                q_update = reward
                q_values = np.zeros(self.action_space).reshape(1, -1)

            q_values[0][action] = q_update

            X.append(state)
            targets.append(q_values[0])
        # print(X)
        # print(targets)
        self.classifier.fit(X, targets)
        self.isFit = True
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
