import numpy as np
import random
import numpy as np
from .utils import predict_input

# from sklearn.ensemble import RandomForestClassifier
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

        # self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
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
        targets = []
        for state, action, state_next, reward in batch:
            if action != None:
                q_update = reward

                if self.isFit and isinstance(state_next, list):
                    q_update = reward + GAMMA * np.amax(
                        self.classifier.predict(predict_input(state_next))[0]
                    )
                    q_values = self.classifier.predict(predict_input(state))
                elif self.isFit and state_next == None:
                    q_values = self.classifier.predict(predict_input(state))
                    q_update = reward
                else:
                    q_update = reward
                    q_values = np.zeros(self.action_space).reshape(1, -1)

                q_values[0][action] = q_update

                X.append(state)
                targets.append(q_values)

        targets = np.argmax(targets, axis=1)
        self.classifier.fit(X, targets)
        self.isFit = True
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
