import events as e
from agent_code.auto_bomber import custom_events as ce


MODELS_ROOT = "./models"
# MODEL_DIR = "./models/41"
MODEL_DIR = None
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPSILON = 0.25
DISCOUNT = 0.5
LEARNING_RATE = 0.1
POLICY = 'IANN'
TEMPERATURE = 0.5
REGION_SIZE = 2
REGION_TIME_TOLERANCE = 8

game_rewards = {
    e.CRATE_DESTROYED: 10,
    e.BOMB_DROPPED: 20,
    e.COIN_FOUND: 10,
    e.COIN_COLLECTED: 50,
    e.KILLED_OPPONENT: 200,
    e.INVALID_ACTION: -1,
    e.KILLED_SELF: -300,
    e.GOT_KILLED: -200,
    e.SURVIVED_ROUND: 300,
    ce.SAME_REGION: -20
}
