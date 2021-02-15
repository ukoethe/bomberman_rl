import pickle
from time import sleep

import numpy as np

import settings as s
from agents import Agent
from environment import GenericWorld, WorldArgs
from fallbacks import pygame
from items import Coin


class ReplayWorld(GenericWorld):
    def __init__(self, args: WorldArgs):
        super().__init__(args)

        replay_file = args.replay
        self.logger.info(f'Loading replay file "{replay_file}"')
        self.replay_file = replay_file
        with open(replay_file, 'rb') as f:
            self.replay = pickle.load(f)
        if not 'n_steps' in self.replay:
            self.replay['n_steps'] = s.MAX_STEPS

        pygame.display.set_caption(f'{replay_file}')

        # Recreate the agents
        self.agents = [ReplayAgent(name, self.colors.pop())
                       for (name, s, b, xy) in self.replay['agents']]
        self.new_round()

    def new_round(self):
        self.logger.info('STARTING REPLAY')

        # Bookkeeping
        self.step = 0
        self.bombs = []
        self.explosions = []
        self.running = True
        self.frame = 0

        # Game world and objects
        self.arena = np.array(self.replay['arena'])
        self.coins = []
        for xy in self.replay['coins']:
            if self.arena[xy] == 0:
                self.coins.append(Coin(xy, True))
            else:
                self.coins.append(Coin(xy, False))
        self.active_agents = [a for a in self.agents]
        for i, agent in enumerate(self.agents):
            agent.start_round()
            agent.x, agent.y = self.replay['agents'][i][-1]
            agent.total_score = 0

    def poll_and_run_agents(self):
        # Perform recorded agent actions
        perm = self.replay['permutations'][self.step - 1]
        for i in perm:
            a = self.active_agents[i]
            self.logger.debug(f'Repeating action from agent <{a.name}>')
            action = self.replay['actions'][a.name][self.step - 1]
            self.logger.info(f'Agent <{a.name}> chose action {action}.')
            self.perform_agent_action(a, action)

    def time_to_stop(self):
        time_to_stop = super().time_to_stop()
        if self.step == self.replay['n_steps']:
            self.logger.info('Replay ends here, wrap up round')
            time_to_stop = True
        return time_to_stop

    def end_round(self):
        super().end_round()
        if self.running:
            self.running = False
            # Wait in case there is still a game step running
            sleep(self.args.update_interval)
        else:
            self.logger.warning('End-of-round requested while no round was running')

        self.logger.debug('Setting ready_for_restart_flag')
        self.ready_for_restart_flag.set()


class ReplayAgent(Agent):
    """
    Agents class firing off a predefined sequence of actions.
    """

    def __init__(self, name, color):
        """Recreate the agent as it was at the beginning of the original game."""
        super().__init__(color, name, None, False, None)

    def setup(self):
        pass

    def act(self, game_state):
        pass

    def wait_for_act(self):
        return 0, self.actions.popleft()
