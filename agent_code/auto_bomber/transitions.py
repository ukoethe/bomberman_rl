import numpy as np


class Transitions:
    def __init__(self, feature_extractor):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []

        self.feature_extractor = feature_extractor

    def add_transition(self, old_game_state, action, new_game_state, rewards):
        self.states.append(self.feature_extractor(old_game_state))
        self.actions.append(action)
        self.next_states.append(self.feature_extractor(new_game_state))
        self.rewards.append(rewards)

    def to_numpy_transitions(self, hyper_parameters):
        return NumpyTransitions(self, hyper_parameters)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        self.rewards.clear()


class NumpyTransitions:
    # todo add hyperparam for batch size to support TD-n-step and monte-carlo
    def __init__(self, transitions, hyper_parameters):
        self.states = np.asarray(transitions.states, dtype=np.float32)
        self.actions = np.asarray(transitions.actions)
        self.next_states = np.asarray(transitions.next_states, dtype=np.float32)
        self.rewards = np.asarray(transitions.rewards, dtype=np.float32)
        self.hyper_parameters = hyper_parameters

    def get_time_steps_for_action(self, action):
        return np.argwhere(self.actions == action)

    def get_features_and_value_estimates(self, action):
        relevant_indices = self.get_time_steps_for_action(action)
        value_estimations = np.zeros((len(relevant_indices),), dtype=np.float32)
        for i in range(len(relevant_indices)):
            value_estimations[i] = self.monte_carlo_value_estimation(np.asscalar(relevant_indices[i]))
        return np.take(self.states, relevant_indices, axis=0).squeeze(axis=1), value_estimations

    def monte_carlo_value_estimation(self, time_step_start: int):
        relevant_rewards = self.rewards[time_step_start:]
        discounts = np.fromfunction(lambda i: self.hyper_parameters["discount"] ** i,
                                    shape=(len(relevant_rewards),), dtype=np.float32)
        return np.sum(discounts * relevant_rewards)
