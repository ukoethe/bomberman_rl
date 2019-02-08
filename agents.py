
from time import time, sleep
import os, signal
from types import SimpleNamespace
import multiprocessing as mp
import importlib
import logging
import pygame
from pygame.locals import *
from pygame.transform import smoothscale

from items import *
from settings import s, e


class IgnoreKeyboardInterrupt(object):
    """Context manager that protects enclosed code from Interrupt signals."""
    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
    def handler(self, sig, frame):
        pass
    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)


class AgentProcess(mp.Process):
    """Wrapper class that runs custom agent code in a separate process."""

    def __init__(self, pipe_to_world, ready_flag, name, agent_dir, train_flag):
        super(AgentProcess, self).__init__(name=name)
        self.pipe_to_world = pipe_to_world
        self.ready_flag = ready_flag
        self.agent_dir = agent_dir
        self.train_flag = train_flag

    def run(self):
        # Persistent 'self' object to pass to callback methods
        self.fake_self = SimpleNamespace(name=self.name)

        # Set up individual loggers for the wrapper and the custom code
        self.wlogger = logging.getLogger(self.name + '_wrapper')
        self.wlogger.setLevel(s.log_agent_wrapper)
        self.fake_self.logger = logging.getLogger(self.name + '_code')
        self.fake_self.logger.setLevel(s.log_agent_code)
        log_dir = f'agent_code/{self.agent_dir}/logs/'
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        handler = logging.FileHandler(f'{log_dir}{self.name}.log', mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.wlogger.addHandler(handler)
        self.fake_self.logger.addHandler(handler)

        # Import custom code for the agent from provided script
        self.wlogger.info(f'Import agent code from "agent_code/{self.agent_dir}/callbacks.py"')
        self.code = importlib.import_module('agent_code.' + self.agent_dir + '.callbacks')

        # Initialize custom code
        self.wlogger.info('Initialize agent code')
        try:
            self.code.setup(self.fake_self)
        except Exception as e:
            self.wlogger.exception(f'Error in callback function: {e}')
        self.wlogger.debug('Set flag to indicate readiness')
        self.ready_flag.set()

        # Play one game after the other until global exit message is received
        while True:
            # Receive round number and check for exit message
            self.wlogger.debug('Wait for new round')
            self.round = self.pipe_to_world.recv()
            if self.round is None:
                self.wlogger.info('Received global exit message')
                break
            self.wlogger.info(f'STARTING ROUND #{self.round}')

            # Take steps until exit message for current round is received
            while True:
                # Receive new game state and check for exit message
                self.wlogger.debug('Receive game state')
                self.fake_self.game_state = self.pipe_to_world.recv()
                if self.fake_self.game_state['exit']:
                    self.ready_flag.set()
                    self.wlogger.info('Received exit message for round')
                    break
                self.wlogger.info(f'STARTING STEP {self.fake_self.game_state["step"]}')

                # Process game events for rewards if in training mode
                if self.train_flag.is_set():
                    self.wlogger.debug('Receive event queue')
                    self.fake_self.events = self.pipe_to_world.recv()
                    self.wlogger.debug(f'Received event queue {self.fake_self.events}')
                    self.wlogger.info('Process intermediate rewards')
                    try:
                        self.code.reward_update(self.fake_self)
                    except Exception as e:
                        self.wlogger.exception(f'Error in callback function: {e}')
                    self.wlogger.debug('Set flag to indicate readiness')
                    self.ready_flag.set()

                # Come up with an action to perform
                self.wlogger.debug('Begin choosing an action')
                self.fake_self.next_action = 'WAIT'
                t = time()
                try:
                    self.code.act(self.fake_self)
                except KeyboardInterrupt:
                    self.wlogger.warn(f'Got interrupted by timeout')
                except Exception as e:
                    self.wlogger.exception(f'Error in callback function: {e}')

                # Send action and time taken back to main process
                with IgnoreKeyboardInterrupt():
                    t = time() - t
                    self.wlogger.info(f'Chose action {self.fake_self.next_action} after {t:.3f}s of thinking')
                    self.wlogger.debug('Send action and time to main process')
                    self.pipe_to_world.send((self.fake_self.next_action, t))
                    while self.ready_flag.is_set():
                        sleep(0.01)
                    self.wlogger.debug('Set flag to indicate readiness')
                    self.ready_flag.set()

            # Process final events and learn from episode if in training mode
            if self.train_flag.is_set():
                self.wlogger.info('Finalize agent\'s training')
                self.wlogger.debug('Receive final event queue')
                self.fake_self.events = self.pipe_to_world.recv()
                self.wlogger.debug(f'Received final event queue {self.fake_self.events}')
                try:
                    self.code.end_of_episode(self.fake_self)
                except Exception as e:
                    self.wlogger.exception(f'Error in callback function: {e}')
                self.ready_flag.set()

            self.wlogger.info(f'Round #{self.round} finished')

        self.wlogger.info('SHUT DOWN')


class Agent(object):
    """Class representing agents as game objects."""

    coin_trophy = smoothscale(pygame.image.load('assets/coin.png'), (15,15))
    suicide_trophy = smoothscale(pygame.image.load('assets/explosion_2.png'), (15,15))
    time_trophy = pygame.image.load('assets/hourglass.png')

    def __init__(self, process, pipe_to_agent, ready_flag, color, train_flag):
        """Set up agent, process for custom code and inter-process communication."""
        self.name = process.name
        self.process = process
        self.pipe = pipe_to_agent
        self.ready_flag = ready_flag
        self.color = color
        self.train_flag = train_flag

        # Load custom avatar or standard robot avatar of assigned color
        try:
            self.avatar = pygame.image.load(f'agent_code/{self.process.agent_dir}/avatar.png')
            assert self.avatar.get_size() == (30,30)
        except Exception as e:
            self.avatar = pygame.image.load(f'assets/robot_{self.color}.png')
        # Load custom bomb sprite
        try:
            self.bomb_sprite = pygame.image.load(f'agent_code/{self.process.agent_dir}/bomb.png')
            assert self.bomb_sprite.get_size() == (30,30)
        except Exception as e:
            self.bomb_sprite = None

        # Prepare overlay that will indicate dead agent on the scoreboard
        self.shade = pygame.Surface((30,30), SRCALPHA)
        self.shade.fill((0,0,0,208))

        self.x, self.y = 1, 1
        self.total_score = 0
        self.bomb_timer = s.bomb_timer + 1
        self.explosion_timer = s.explosion_timer + 1
        self.bomb_power = s.bomb_power
        self.bomb_type = Bomb

        self.reset()

    def reset(self, current_round=None):
        """Make agent ready for a new game round."""
        if current_round:
            self.pipe.send(current_round)
        self.times = []
        self.mean_time = 0
        self.dead = False
        self.score = 0
        self.events = []
        self.bombs_left = 1
        self.trophies = []

    def get_state(self):
        """Provide information about this agent for the global game state."""
        return (self.x, self.y, self.name, self.bombs_left)

    def update_score(self, delta):
        """Add delta to both the current round's score and the total score."""
        self.score += delta
        self.total_score += delta

    def make_bomb(self):
        """Create a new Bomb object at current agent position."""
        return self.bomb_type((self.x, self.y), self,
                              self.bomb_timer, self.bomb_power, self.color,
                              custom_sprite=self.bomb_sprite)

    def render(self, screen, x, y):
        """Draw the agent's avatar to the screen at the given coordinates."""
        screen.blit(self.avatar, (x, y))
        if self.dead:
            screen.blit(self.shade, (x, y))



class ReplayAgent(Agent):
    """Agents class specifically for playing back pre-recorded games."""

    def __init__(self, name, color, x, y):
        """Recreate the agent as it was at the beginning of the original game."""
        self.name = name
        self.x, self.y = x, y
        self.color = color

        # Load custom avatar or standard robot avatar of assigned color
        self.avatar = pygame.image.load(f'assets/robot_{self.color}.png')
        # Prepare overlay that will indicate dead agent on the scoreboard
        self.shade = pygame.Surface((30,30), SRCALPHA)
        self.shade.fill((0,0,0,208))

        self.total_score = 0
        self.bomb_timer = s.bomb_timer
        self.explosion_timer = s.explosion_timer
        self.bomb_power = s.bomb_power
        self.bomb_type = Bomb

        self.reset()
