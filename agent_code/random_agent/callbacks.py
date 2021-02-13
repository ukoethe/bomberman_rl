import numpy as np


def setup(self):
    np.random.seed()


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random')
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
