import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter

import agent_code.auto_bomber.model_path as model_path
from agent_code.auto_bomber.transitions import Transitions


def get_model_dir():
    try:
        return os.environ["MODEL_DIR"]
    except KeyError as e:
        return model_path.MODEL_DIR


class LinearAutoBomberModel:
    def __init__(self, train, feature_extractor):
        self.train = train
        self.weights = None
        self.feature_extractor = feature_extractor

        model_dir = get_model_dir()
        if model_dir and Path(model_dir).is_dir():
            self.model_dir = Path(model_dir)
        elif model_dir and not Path(model_dir).is_dir():
            raise FileNotFoundError("The specified model directory does not exist!\nIf you wish to train a NEW model"
                                    "set parameter to None, otherwise specify a valid model directory.")
        elif not self.train and not model_dir:
            raise ValueError("No model directory has been specified.\n A model directory is required for inference.")
        else:
            root_dir = Path(model_path.MODELS_ROOT)
            root_dir.mkdir(parents=True, exist_ok=True)
            existing_subdirs = sorted([int(x.stem) for x in root_dir.iterdir() if x.is_dir()])

            model_index = existing_subdirs[-1] if existing_subdirs else -1
            model_index += 1
            self.model_dir = Path(model_path.MODELS_ROOT) / str(model_index)
            self.model_dir.mkdir()
            # Copy configuration file for logging purposes
            shutil.copy(Path("default_hyper_parameters.json"), self.model_dir / "hyper_parameters.json")

        self.weights_path = self.model_dir / "weights.pt"
        if self.weights_path.is_file():
            with self.weights_path.open(mode="rb") as file:
                self.weights = pickle.load(file)

        hyper_parameters_path = self.model_dir / "hyper_parameters.json"
        if hyper_parameters_path.is_file():
            with hyper_parameters_path.open(mode="rb") as file:
                self.hyper_parameters = json.load(file)

        if self.train:
            self.writer = SummaryWriter(logdir=f"./runs/exp{self.model_dir.stem}")

    def store(self):
        with self.weights_path.open(mode="wb") as file:
            pickle.dump(self.weights, file)

    def select_best_action(self, game_state: dict, agent_self, softmax=False):
        features_x = self.feature_extractor(game_state)
        self.init_if_needed(features_x, agent_self)

        q_action_values = np.dot(self.weights, features_x)

        if softmax:
            temp = self.hyper_parameters["temperature"]
            p = np.exp(q_action_values / temp) / np.sum(np.exp(q_action_values / temp))
            choice = np.random.choice(len(q_action_values), p=p)
        else:
            top_3_actions = q_action_values.argsort()[-3:][::-1]
            choice = np.random.choice(top_3_actions, p=[0.9, 0.05, 0.05])
        return self.hyper_parameters["actions"][choice]

    def fit_model_with_transition_batch(self, transitions: Transitions, round: int):
        loss = []
        numpy_transitions = transitions.to_numpy_transitions(self.hyper_parameters)
        for action_id, action in enumerate(self.hyper_parameters["actions"]):
            x_all_t, y_all_t = numpy_transitions.get_features_and_value_estimates(action)

            if x_all_t.size != 0:
                q_estimations = np.dot(x_all_t, self.weights[action_id])
                residuals = (y_all_t - q_estimations)
                loss.append(np.mean(residuals ** 2))
                q_grad = np.dot(x_all_t.T, residuals)

                weight_updates = self.hyper_parameters["learning_rate"] / y_all_t.shape[0] * q_grad
                self.weights[action_id] += weight_updates

        mean_loss = np.mean(loss)
        self.writer.add_scalar('loss', mean_loss, round)
        mean_reward = np.mean(numpy_transitions.rewards)
        self.writer.add_scalar('rewards', mean_reward, round)

    def init_if_needed(self, features_x, agent_self):
        if self.weights is None:
            agent_self.logger.info("Model is empty init with random weights.")

            # Xavier weights initialization
            self.weights = np.random.rand(len(self.hyper_parameters["actions"]),
                                          len(features_x)) * np.sqrt(1 / len(features_x))
