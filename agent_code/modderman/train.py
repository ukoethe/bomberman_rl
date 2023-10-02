from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np

import events as e
from .features_actual import BombermanFeatures


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INT = {'UP':0, 'RIGHT' : 1 , 'DOWN': 2, 'LEFT': 3, 'WAIT':4, 'BOMB': 5}

FEATURE_MAP = {"DIRECTION_TO_TARGET": [0,1,2,3],
                   "GOOD_BOMB_SPOT": 4,
                   "DANGEROUS_ACTION": [5,6,7,8,9],
                   "EXPLOSION_NEARBY": [10,11,12,13],
                   "VALID_MOVES": [14,15,16,17],
                   "BOMB_ACTIVE": 18}
NUM_FEATURES = 19

# Data model for the step-wise transitions within rounds
StepTransition = namedtuple('StepTransition',
                        ('state', 'action', 'next_state', 'reward', 'stepCount'))

# Data model for the transitions between rounds
RoundTransition = namedtuple('RoundTransition', ('state', 'action', 'next_state', 'total_round_rewards', 'stepCount', 'total_round_next_states'))

# Hyper parameters -- DO modify
GAMMA = 0.2 # Discount factor to be used in Q value updates
TEMPORAL_DIFFERENCE_STEP = 3 # How many steps within a round to look to in the future for temporal difference calculations

TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 800 # Batch size of subset used for gradient updates
BATCH_PRIORITY_SIZE = 100

MAX_STEPS = 400


def monte_carlo_prediction_scheme(self, total_round_rewards, current_time_step):
    discounted_return = 0
    for time_step_offset, time_step in enumerate(range(current_time_step - 1, len(total_round_rewards))):
        discounted_return += np.power(GAMMA, time_step_offset) * total_round_rewards[time_step]
    return discounted_return


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.feature_extractor = BombermanFeatures()
    self.round_reward = 0

    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) # Last TRANSITION_HISTORY_SIZE transitions including both round and step-wise transitions
    self.current_round_transitions = deque(maxlen=MAX_STEPS) # All transitions so far in the current round
    self.current_round_rewards = deque(maxlen=MAX_STEPS) # All rewards collected step-wise in the current round
    self.current_round_next_states = deque(maxlen=MAX_STEPS) # All next states achieved in the current round so far

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state == None:
        return

    events = append_custom_events(self, old_game_state, new_game_state, events)
    stepwise_reward = reward_from_events(self, events)
    self.round_reward = self.round_reward + stepwise_reward


    # state_to_features is defined in callbacks.py
    self.current_round_transitions.append(StepTransition(self.feature_extractor.state_to_features(old_game_state), self_action, self.feature_extractor.state_to_features(new_game_state), reward_from_events(self, events), old_game_state['step']))
    self.current_round_rewards.append(reward_from_events(self, events))
    self.current_round_next_states.append(self.feature_extractor.state_to_features(new_game_state))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

     #append last transition
    self.current_round_transitions.append(StepTransition(self.feature_extractor.state_to_features(last_game_state), last_action, None, reward_from_events(self, last_game_state), last_game_state['step']))
    self.current_round_rewards.append(reward_from_events(self, last_game_state))
    self.current_round_next_states.append(None)

    for _, transition in enumerate(self.current_round_transitions):
        self.transitions.append(RoundTransition(transition.state, transition.action, transition.next_state, self.current_round_rewards.copy(), transition.stepCount, self.current_round_next_states.copy()))
    
    # Reset round-wise values
    self.total_rewards = 0
    self.current_round_transitions.clear()
    self.current_round_next_states.clear()
    self.current_round_rewards.clear()

    if last_game_state['round'] < 49:
        return

    # Randomly select indices from transitions
    sample_indices = np.random.choice(np.arange(len(self.transitions), dtype=int), min(len(self.transitions), BATCH_SIZE), replace=False)

    # Create a batch from selected indices
    batch = np.array(self.transitions, dtype=RoundTransition)[sample_indices]

    # Iterate through possible actions
    for action in ACTIONS:
        # Filter subbatch for the current action
        subbatch_indices = np.where(batch[:, 1] == action)[0]
        subbatch = batch[subbatch_indices]

        # Create an array to store old states in the subbatch
        subbatch_old_states = np.array([transition[0] for transition in subbatch])

        # Check if subbatch is not empty
        if len(subbatch) != 0:
            # Use a Monte Carlo prediction scheme to compute approximated Q-values for each action
            if last_game_state['round'] == 49:
                response = np.array([monte_carlo_prediction_scheme(self, transition[3], transition[4]) for transition in subbatch])
            else:
                response = np.array([monte_carlo_prediction_scheme(self, transition[3], transition[4]) for transition in subbatch])

            # Fit decision trees for the current action
            self.decision_trees[ACTION_TO_INT[action]].fit(subbatch_old_states, response)

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.decision_trees, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -80,
        #e.KILLED_SELF: -500,
        e.GOT_KILLED: -500,
        'LIFE_SAVING_MOVE': 20,
        'GOOD_BOMB_PLACEMENT': 10,
        'BAD_BOMB_PLACEMENT': -50,
        'DEADLY_MOVE': -150,
        'MOVES_TOWARD_TARGET': 5,
        'WAITING_ONLY_OPTION': 10,
        'BAD_MOVE': -4,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def append_custom_events(self, old_game_state, new_game_state, events):

    features = self.feature_extractor.state_to_features(old_game_state)

    _, _, _, old_pos = old_game_state['self']
    _, _, _, new_pos =  new_game_state['self']

    danger_left , danger_right, danger_up, danger_down , danger_wait = features[FEATURE_MAP['DANGEROUS_ACTION']]

    if e.INVALID_ACTION in events:
        return events

    #check, if waiting is dangerous we need to move 
    if danger_wait == 1: 
        #check if did a life saving move
        if danger_left == 0 and e.MOVED_LEFT in events:
            events.append("LIFE_SAVING_MOVE")
        elif danger_right == 0 and e.MOVED_RIGHT in events:
            events.append("LIFE_SAVING_MOVE")
        elif danger_up == 0 and e.MOVED_UP in events:
            events.append("LIFE_SAVING_MOVE")
        elif danger_down == 0 and e.MOVED_DOWN in events:
            events.append("LIFE_SAVING_MOVE")
        else: 
            events.append("DEADLY_MOVE")

    elif e.BOMB_DROPPED in events:
        #check if dropped the bomb correctly
        if features[FEATURE_MAP['GOOD_BOMB_SPOT']] == 1:
            events.append("GOOD_BOMB_PLACEMENT")
        else:
            events.append("BAD_BOMB_PLACEMENT")
    else:
        valid_list = features[FEATURE_MAP['VALID_MOVES']].copy()
        valid_list[ np.where( np.logical_or(features[FEATURE_MAP['DANGEROUS_ACTION']][0:4] == 1, features[FEATURE_MAP['EXPLOSION_NEARBY']] == 1) ) ] = 0

        explosion_left , explosion_right, explosion_up, explosion_down = features[FEATURE_MAP['EXPLOSION_NEARBY']]
        target_left , target_right, target_up, target_down = features[FEATURE_MAP['DIRECTION_TO_TARGET']]
        
        
        if np.all(valid_list == 0) and e.WAITED in events:
            events.append("WAITING_ONLY_OPTION")
        
        #check if performed a deadly move 
        #->bomb
        elif (danger_left == 1 and e.MOVED_LEFT in events) or (danger_right == 1 and e.MOVED_RIGHT in events) or (danger_up == 1 and e.MOVED_UP in events) or (danger_down == 1 and e.MOVED_DOWN in events) or (danger_wait == 1 and e.WAITED in events):
            events.append("DEADLY_MOVE")
        #->explosion
        elif (explosion_left == 1 and e.MOVED_LEFT in events) or (explosion_right == 1 and e.MOVED_RIGHT in events) or (explosion_up== 1 and e.MOVED_UP in events) or (explosion_down== 1 and e.MOVED_DOWN in events):
            events.append("DEADLY_MOVE")

        #check if move towards a target
        #->coin and crate as well as opponent
        elif (target_left == 1 and e.MOVED_LEFT in events) or ( target_right == 1 and e.MOVED_RIGHT in events) or ( target_up == 1 and e.MOVED_UP in events) or ( target_down and e.MOVED_DOWN in events):
            events.append("MOVES_TOWARD_TARGET")
        else:
            events.append("BAD_MOVE")

    return events