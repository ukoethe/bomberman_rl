from unittest import TestCase
from unittest.mock import patch, mock_open, Mock
import numpy as np
import pickle

import os
from agent_code.auto_bomber.model import LinearAutoBomberModel


class TestLinearAutoBomberModel(TestCase):

    @patch.object(os.path, attribute="isfile")
    def test_select_best_action(self, mock_path):
        mock_path.isfile.return_value = True

        # get rid of randomness for testing here
        np.random.seed(0)

        read_array = pickle.dumps(np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.float32))
        open_mock = mock_open(read_data=read_array)
        with patch('builtins.open', open_mock):
            feature_array = np.array([3, 2, 1], dtype=np.float32)
            elem = LinearAutoBomberModel(lambda x: feature_array)
            action = elem.select_best_action({}, None)

            self.assertEqual("UP", action)
