import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from typing import List
from .features_actual import BombermanFeatures
from sklearn.tree import DecisionTreeRegressor

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INT = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}
NUM_FEATURES = 19

# Data model for the step-wise transitions within rounds
StepTransition = namedtuple('StepTransition', ('state', 'action', 'next_state', 'reward', 'stepCount'))

# Data model for the transitions between rounds
RoundTransition = namedtuple('RoundTransition', ('state', 'action', 'next_state', 'total_round_rewards', 'stepCount', 'total_round_next_states'))

# Hyperparameters
GAMMA = 0.2  # Discount factor
TEMPORAL_DIFFERENCE_STEP = 3
TRANSITION_HISTORY_SIZE = 1000
RECORD_ENEMY_TRANSITIONS = 1.0
BATCH_SIZE = 800
BATCH_PRIORITY_SIZE = 100
MAX_STEPS = 400
EPSILON = 0.2  # Exploration probability
ALPHA = 0.1  # Learning rate
NUM_EPISODES = 100  # Number of training episodes
MAX_ROUNDS = 49  # Maximum number of rounds per episode

def monte_carlo_prediction_scheme(agent, total_round_rewards, current_time_step):
    discounted_return = 0
    for time_step_offset, time_step in enumerate(range(current_time_step - 1, len(total_round_rewards))):
        discounted_return += np.power(GAMMA, time_step_offset) * total_round_rewards[time_step]
    return discounted_return

def setup_training(agent):
    agent.feature_extractor = BombermanFeatures()
    agent.round_reward = 0
    agent.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    agent.current_round_transitions = deque(maxlen=MAX_STEPS)
    agent.current_round_rewards = deque(maxlen=MAX_STEPS)
    agent.current_round_next_states = deque(maxlen=MAX_STEPS)

def game_events_occurred(agent, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    events = append_custom_events(agent, old_game_state, new_game_state, events)
    stepwise_reward = reward_from_events(agent, events)
    agent.round_reward = agent.round_reward + stepwise_reward

    agent.current_round_transitions.append(StepTransition(agent.feature_extractor.state_to_features(old_game_state), self_action, agent.feature_extractor.state_to_features(new_game_state), reward_from_events(agent, events), old_game_state['step']))
    agent.current_round_rewards.append(reward_from_events(agent, events))
    agent.current_round_next_states.append(agent.feature_extractor.state_to_features(new_game_state))

def end_of_round(agent, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state['round'] < 49:
        return

    # SARSA Update
    Q_values = []

    for t in range(len(agent.current_round_transitions) - 1):
        state_t = agent.current_round_transitions[t].state
        action_t = agent.current_round_transitions[t].action
        reward_t = agent.current_round_rewards[t]
        state_t1 = agent.current_round_transitions[t + 1].state
        action_t1 = agent.current_round_transitions[t + 1].action

        Q_t = agent.decision_trees[ACTION_TO_INT[action_t]].predict([state_t])[0]
        Q_t1 = agent.decision_trees[ACTION_TO_INT[action_t1]].predict([state_t1])[0]

        Q_t += ALPHA * (reward_t + GAMMA * Q_t1 - Q_t)
        Q_values.append(QValue(state_t, action_t, Q_t))

    for q_value in Q_values:
        agent.decision_trees[ACTION_TO_INT[q_value.action]].fit([q_value.state], [q_value.value])

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(agent.decision_trees, file)

    agent.total_rewards = 0
    agent.current_round_transitions.clear()
    agent.current_round_next_states.clear()
    agent.current_round_rewards.clear()

def reward_from_events(agent, events: List[str]) -> int:
    game_rewards = {
        e.INVALID_ACTION: -80,
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
    agent.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def append_custom_events(agent, old_game_state, new_game_state, events):
    features = agent.feature_extractor.state_to_features(old_game_state)
    _, _, _, old_pos = old_game_state['self']
    _, _, _, new_pos = new_game_state['self']
    danger_left, danger_right, danger_up, danger_down, danger_wait = features[FEATURE_MAP['DANGEROUS_ACTION']]

    if e.INVALID_ACTION in events:
        return events

    if danger_wait == 1:
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
        if features[FEATURE_MAP['GOOD_BOMB_SPOT']] == 1:
            events.append("GOOD_BOMB_PLACEMENT")
        else:
            events.append("BAD_BOMB_PLACEMENT")
    else:
        valid_list = features[FEATURE_MAP['VALID_MOVES']].copy()
        valid_list[np.where(np.logical_or(features[FEATURE_MAP['DANGEROUS_ACTION']][0:4] == 1, features[FEATURE_MAP['EXPLOSION_NEARBY']] == 1))] = 0
        explosion_left, explosion_right, explosion_up, explosion_down = features[FEATURE_MAP['EXPLOSION_NEARBY']]
        target_left, target_right, target_up, target_down = features[FEATURE_MAP['DIRECTION_TO_TARGET']]

        if np.all(valid_list == 0) and e.WAITED in events:
            events.append("WAITING_ONLY_OPTION")
        elif (danger_left == 1 and e.MOVED_LEFT in events) or (danger_right == 1 and e.MOVED_RIGHT in events) or \
                (danger_up == 1 and e.MOVED_UP in events) or (danger_down == 1 and e.MOVED_DOWN in events) or \
                (danger_wait == 1 and e.WAITED in events):
            events.append("DEADLY_MOVE")
        elif (explosion_left == 1 and e.MOVED_LEFT in events) or (explosion_right == 1 and e.MOVED_RIGHT in events) or \
                (explosion_up == 1 and e.MOVED_UP in events) or (explosion_down == 1 and e.MOVED_DOWN in events):
            events.append("DEADLY_MOVE")
        elif (target_left == 1 and e.MOVED_LEFT in events) or (target_right == 1 and e.MOVED_RIGHT in events) or \
                (target_up == 1 and e.MOVED_UP in events) or (target_down and e.MOVED_DOWN in events):
            events.append("MOVES_TOWARD_TARGET")
        else:
            events.append("BAD_MOVE")

    return events

def plot_rewards(rewards):
    plt.figure(figsize=(8, 6))
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Round Rewards")
    plt.title("Training Progress")
    plt.grid(True)
    plt.show()

def train_agent():
    agent = Agent()  # Initialize your agent
    setup_training(agent)
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        # Environment setup for each episode
        # ...

        while not game_state['round'] > MAX_ROUNDS:
            # ...

        episode_rewards.append(agent.round_reward)

    plot_rewards(episode_rewards)

if __name__ == "__main__":
    train_agent()
