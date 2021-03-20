import events as e
from agent_code.auto_bomber import custom_events as ce

MODEL_PATH = "./auto_bomber_weights.pt"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPSILON = 0.25
DISCOUNT = 0.5
LEARNING_RATE = 0.1
REGION_SIZE = 2
REGION_TIME_TOLERANCE = 8

game_rewards = {
e.COIN_COLLECTED: 50,
e.KILLED_OPPONENT: 75,
e.INVALID_ACTION: -1,
e.KILLED_SELF: -100,
e.GOT_KILLED: -75,
e.SURVIVED_ROUND: 1000,
ce.UP_AND_DOWN: -50,
ce.LEFT_AND_RIGHT: -50,
ce.SAME_REGION: -20
}