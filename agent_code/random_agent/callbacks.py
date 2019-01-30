
import numpy as np


def setup(agent):
    np.random.seed()

def act(agent):
    agent.logger.info('Pick action at random')
    agent.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])

def reward_update(agent):
    pass

def end_of_episode(agent):
    pass
