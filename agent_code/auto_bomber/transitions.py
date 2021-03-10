import numpy as np
import agent_code.auto_bomber.auto_bomber_config as config


class Transitions:
    def __init__(self, feature_extractor):
        self.actions = []
        self.states = []
        self.next_states = []
        self.rewards = []

        self.feature_extractor = feature_extractor

    def add_transition(self, action, old_game_state, new_game_state, rewards):
        self.actions.append(action)
        self.states.append(self.feature_extractor(old_game_state))
        self.next_states.append(self.feature_extractor(new_game_state))
        self.rewards.append(rewards)

    def to_numpy_transitions(self):
        return NumpyTransitions(self)


class NumpyTransitions:
    def __init__(self, transitions):
        self.actions = np.array(transitions.actions)
        self.states = np.array(transitions.states, dtype=np.float32)
        self.next_states = np.array(transitions.next_states, dtype=np.float32)
        self.rewards = np.array(transitions.rewards, dtype=np.float32)

    def get_time_steps_for_action(self, action):
        return np.argwhere(self.actions == action)

    # todo test
    def value_estimation_vector_for_action(self, action):
        relevant_indices = self.get_time_steps_for_action(action)
        value_estimations = np.zeros((len(relevant_indices),), dtype=np.float32)
        for i in range(len(relevant_indices)):
            value_estimations[i] = self.monte_carlo_value_estimation(relevant_indices[i])
        return value_estimations

    def monte_carlo_value_estimation(self, time_step: int):
        relevant_rewards = self.rewards[time_step:]
        discounts = np.fromfunction(lambda i: config.DISCOUNT ** i, shape=(len(relevant_rewards),), dtype=np.float32)
        return np.sum(discounts * relevant_rewards)

