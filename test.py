import os
import unittest
from time import time

from main import main


class MainTestCase(unittest.TestCase):
    def test_play(self):
        start_time = time()
        main(["play", "--n-rounds", "1", "--no-gui"])
        # Assert that log exists
        self.assertTrue(os.path.isfile("logs/game.log"))
        # Assert that game log way actually written
        self.assertGreater(os.path.getmtime("logs/game.log"), start_time)


if __name__ == '__main__':
    unittest.main()
