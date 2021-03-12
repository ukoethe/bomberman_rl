import logging
import pickle
import random
from collections import namedtuple
from datetime import datetime
from logging.handlers import RotatingFileHandler
from os.path import dirname
from threading import Event
from time import time
from typing import List, Union

import numpy as np

import events as e
import settings as s
from agents import Agent, SequentialAgentBackend
from fallbacks import pygame
from items import Coin, Explosion, Bomb

WorldArgs = namedtuple("WorldArgs",
                       ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay", "make_video", "continue_without_training", "log_dir"])


class Trophy:
    coin_trophy = pygame.transform.smoothscale(pygame.image.load('assets/coin.png'), (15, 15))
    suicide_trophy = pygame.transform.smoothscale(pygame.image.load('assets/explosion_2.png'), (15, 15))
    time_trophy = pygame.image.load('assets/hourglass.png')


class GenericWorld:
    logger: logging.Logger

    running: bool = False
    step: int

    agents: List[Agent]
    active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]

    gui: Union[None, "GUI"]
    round_id: str

    def __init__(self, args: WorldArgs):
        self.args = args
        self.setup_logging()
        if self.args.no_gui:
            self.gui = None
        else:
            self.gui = GUI(args, self)

        self.colors = s.AGENT_COLORS

        self.round = 0
        self.running = False
        self.ready_for_restart_flag = Event()

    def setup_logging(self):
        self.logger = logging.getLogger('BombeRLeWorld')
        self.logger.setLevel(s.LOG_GAME)
        handler = logging.FileHandler(f'{self.args.log_dir}/game.log', mode="w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Initializing game world')

    def new_round(self):
        raise NotImplementedError()

    def add_agent(self, agent_dir, name, train=False):
        assert len(self.agents) < s.MAX_AGENTS

        # if self.args.single_process:
        backend = SequentialAgentBackend(train, name, agent_dir)
        # else:
        # backend = ProcessAgentBackend(train, name, agent_dir)
        backend.start()

        agent = Agent(self.colors.pop(), name, agent_dir, train, backend)
        self.agents.append(agent)

    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free

    def perform_agent_action(self, agent: Agent, action: str):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.add_event(e.MOVED_UP)
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.add_event(e.MOVED_DOWN)
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.add_event(e.MOVED_LEFT)
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.add_event(e.MOVED_RIGHT)
        elif action == 'BOMB' and agent.bombs_left:
            self.logger.info(f'Agent <{agent.name}> drops bomb at {(agent.x, agent.y)}')
            self.bombs.append(Bomb((agent.x, agent.y), agent, s.BOMB_TIMER, s.BOMB_POWER, agent.color, custom_sprite=agent.bomb_sprite))
            agent.bombs_left = False
            agent.add_event(e.BOMB_DROPPED)
        elif action == 'WAIT':
            agent.add_event(e.WAITED)
        else:
            agent.add_event(e.INVALID_ACTION)

    def poll_and_run_agents(self):
        raise NotImplementedError()

    def do_step(self, user_input='WAIT'):
        self.step += 1
        self.logger.info(f'STARTING STEP {self.step}')

        self.user_input = user_input
        self.logger.debug(f'User input: {self.user_input}')

        self.poll_and_run_agents()

        self.collect_coins()
        self.update_bombs()
        self.evaluate_explosions()

        if self.time_to_stop():
            self.end_round()

    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        self.logger.info(f'Agent <{a.name}> picked up coin at {(a.x, a.y)} and receives 1 point')
                        a.update_score(s.REWARD_COIN)
                        a.add_event(e.COIN_COLLECTED)
                        a.trophies.append(Trophy.coin_trophy)

    def update_bombs(self):
        """
        Count down bombs placed
        Explode bombs at zero timer.

        :return:
        """
        for bomb in self.bombs:
            if bomb.timer <= 0:
                # Explode when timer is finished
                self.logger.info(f'Agent <{bomb.owner.name}>\'s bomb at {(bomb.x, bomb.y)} explodes')
                bomb.owner.add_event(e.BOMB_EXPLODED)
                blast_coords = bomb.get_blast_coords(self.arena)

                # Clear crates
                for (x, y) in blast_coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        bomb.owner.add_event(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x, c.y) == (x, y):
                                c.collectable = True
                                self.logger.info(f'Coin found at {(x, y)}')
                                bomb.owner.add_event(e.COIN_FOUND)

                # Create explosion
                screen_coords = [(s.GRID_OFFSET[0] + s.GRID_SIZE * x, s.GRID_OFFSET[1] + s.GRID_SIZE * y) for (x, y) in
                                 blast_coords]
                self.explosions.append(Explosion(blast_coords, screen_coords, bomb.owner, s.EXPLOSION_TIMER))
                bomb.active = False
                bomb.owner.bombs_left = True
            else:
                # Progress countdown
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]

    def evaluate_explosions(self):
        # Explosions
        agents_hit = set()
        for explosion in self.explosions:
            # Kill agents
            if explosion.timer > 1:
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(f'Agent <{a.name}> blown up by own bomb')
                            a.add_event(e.KILLED_SELF)
                            explosion.owner.trophies.append(Trophy.suicide_trophy)
                        else:
                            self.logger.info(f'Agent <{a.name}> blown up by agent <{explosion.owner.name}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(s.REWARD_KILL)
                            explosion.owner.add_event(e.KILLED_OPPONENT)
                            explosion.owner.trophies.append(pygame.transform.smoothscale(a.avatar, (15, 15)))
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False

            # Progress countdown
            explosion.timer -= 1
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            a.add_event(e.GOT_KILLED)
            for aa in self.active_agents:
                if aa is not a:
                    aa.add_event(e.OPPONENT_ELIMINATED)
        self.explosions = [exp for exp in self.explosions if exp.active]

    def end_round(self):
        # Turn screenshots into videos
        if self.args.make_video:
            self.logger.debug(f'Turning screenshots into video files')
            import subprocess, os, glob
            subprocess.call(['ffmpeg', '-y', '-framerate', f'{self.args.fps}',
                             '-f', 'image2', '-pattern_type', 'glob', '-i', f'screenshots/{self.round_id}_*.png',
                             '-preset', 'veryslow', '-tune', 'animation', '-crf', '5', '-c:v', 'libx264', '-pix_fmt',
                             'yuv420p',
                             f'screenshots/{self.round_id}_video.mp4'])
            subprocess.call(['ffmpeg', '-y', '-framerate', f'{self.args.fps}',
                             '-f', 'image2', '-pattern_type', 'glob', '-i', f'screenshots/{self.round_id}_*.png',
                             '-threads', '2', '-tile-columns', '2', '-frame-parallel', '0', '-g', '100', '-speed', '1',
                             '-pix_fmt', 'yuv420p', '-qmin', '0', '-qmax', '10', '-crf', '5', '-b:v', '2M', '-c:v',
                             'libvpx-vp9',
                             f'screenshots/{self.round_id}_video.webm'])
            for f in glob.glob(f'screenshots/{self.round_id}_*.png'):
                os.remove(f)

    def time_to_stop(self):
        # Check round stopping criteria
        if len(self.active_agents) == 0:
            self.logger.info(f'No agent left alive, wrap up round')
            return True

        if (len(self.active_agents) == 1
                and (self.arena == 1).sum() == 0
                and all([not c.collectable for c in self.coins])
                and len(self.bombs) + len(self.explosions) == 0):
            self.logger.info(f'One agent left alive with nothing to do, wrap up round')
            return True

        if any(a.train for a in self.agents) and not self.args.continue_without_training:
            if not any([a.train for a in self.active_agents]):
                self.logger.info('No training agent left alive, wrap up round')
                return True

        if self.step >= s.MAX_STEPS:
            self.logger.info('Maximum number of steps reached, wrap up round')
            return True

        return False

    def render(self):
        self.gui.render()

        # Save screenshot
        if self.args.make_video:
            self.logger.debug(f'Saving screenshot for frame {self.gui.frame}')
            pygame.image.save(self.gui.screen, dirname(__file__) + f'/screenshots/{self.round_id}_{self.gui.frame:05d}.png')

    def end(self):
        # Turn screenshots into videos
        if self.args.make_video:
            self.logger.debug(f'Turning screenshots into video files')
            import subprocess, os, glob
            subprocess.call(['ffmpeg', '-y', '-framerate', f'{self.args.fps}',
                             '-f', 'image2', '-pattern_type', 'glob', '-i', f'screenshots/{self.round_id}_*.png',
                             '-preset', 'veryslow', '-tune', 'animation', '-crf', '5', '-c:v', 'libx264', '-pix_fmt',
                             'yuv420p',
                             f'screenshots/{self.round_id}_video.mp4'])
            subprocess.call(['ffmpeg', '-y', '-framerate', f'{self.args.fps}',
                             '-f', 'image2', '-pattern_type', 'glob', '-i', f'screenshots/{self.round_id}_*.png',
                             '-threads', '2', '-tile-columns', '2', '-frame-parallel', '0', '-g', '100', '-speed', '1',
                             '-pix_fmt', 'yuv420p', '-qmin', '0', '-qmax', '10', '-crf', '5', '-b:v', '2M', '-c:v',
                             'libvpx-vp9',
                             f'screenshots/{self.round_id}_video.webm'])
            for f in glob.glob(f'screenshots/{self.round_id}_*.png'):
                os.remove(f)


class BombeRLeWorld(GenericWorld):
    def __init__(self, args: WorldArgs, agents):
        super().__init__(args)

        self.setup_agents(agents)
        self.new_round()

    def setup_agents(self, agents):
        # Add specified agents and start their subprocesses
        self.agents = []
        for agent_dir, train in agents:
            if list([d for d, t in agents]).count(agent_dir) > 1:
                name = agent_dir + '_' + str(list([a.code_name for a in self.agents]).count(agent_dir))
            else:
                name = agent_dir
            self.add_agent(agent_dir, name, train=train)

    def new_round(self):
        if self.running:
            self.logger.warning('New round requested while still running')
            self.end_round()

        self.round += 1
        self.logger.info(f'STARTING ROUND #{self.round}')
        pygame.display.set_caption(f'BombeRLe | Round #{self.round}')

        # Bookkeeping
        self.step = 0
        self.active_agents = []
        self.bombs = []
        self.explosions = []
        self.round_id = f'Replay {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'

        # Arena with wall and crate layout
        self.arena = (np.random.rand(s.COLS, s.ROWS) < s.CRATE_DENSITY).astype(int)
        self.arena[:1, :] = -1
        self.arena[-1:, :] = -1
        self.arena[:, :1] = -1
        self.arena[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    self.arena[x, y] = -1

        # Starting positions
        start_positions = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
        random.shuffle(start_positions)
        for (x, y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if self.arena[xx, yy] == 1:
                    self.arena[xx, yy] = 0

        # Distribute coins evenly
        self.coins = []
        """coin_pattern = np.array([
            [1, 1, 1],
            [0, 0, 1],
        ])
        coins = np.zeros_like(self.arena)
        for x in range(1, s.COLS - 2, coin_pattern.shape[0]):
            for i in range(coin_pattern.shape[0]):
                for j in range(coin_pattern.shape[1]):
                    if coin_pattern[i, j] == 1:
                        self.coins.append(Coin((x + i, x + j), self.arena[x+i,x+j] == 0))
                        coins[x + i, x + j] += 1"""
        for i in range(3):
            for j in range(3):
                n_crates = (self.arena[1 + 5 * i:6 + 5 * i, 1 + 5 * j:6 + 5 * j] == 1).sum()
                while True:
                    x, y = np.random.randint(1 + 5 * i, 6 + 5 * i), np.random.randint(1 + 5 * j, 6 + 5 * j)
                    if n_crates == 0 and self.arena[x, y] == 0:
                        self.coins.append(Coin((x, y)))
                        self.coins[-1].collectable = True
                        break
                    elif self.arena[x, y] == 1:
                        self.coins.append(Coin((x, y)))
                        break

        # Reset agents and distribute starting positions
        for agent in self.agents:
            agent.start_round()
            self.active_agents.append(agent)
            agent.x, agent.y = start_positions.pop()

        self.replay = {
            'round': self.round,
            'arena': np.array(self.arena),
            'coins': [c.get_state() for c in self.coins],
            'agents': [a.get_state() for a in self.agents],
            'actions': dict([(a.name, []) for a in self.agents]),
            'permutations': []
        }

        self.running = True

    def get_state_for_agent(self, agent: Agent):
        state = {
            'round': self.round,
            'step': self.step,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'others': [other.get_state() for other in self.active_agents if other is not agent],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
            'user_input': self.user_input,
        }

        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            for (x, y) in exp.blast_coords:
                explosion_map[x, y] = max(explosion_map[x, y], exp.timer)
        state['explosion_map'] = explosion_map

        return state

    def send_training_events(self):
        # Send events to all agents that expect them, then reset and wait for them
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.process_game_events(self.get_state_for_agent(a))
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.process_enemy_game_events(self.get_state_for_agent(enemy), enemy)
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.wait_for_game_event_processing()
                for enemy in self.active_agents:
                    if enemy is not a:
                        pass
                        # a.wait_for_enemy_game_event_processing()
        for a in self.active_agents:
            a.store_game_state(self.get_state_for_agent(a))
            a.reset_game_events()

    def poll_and_run_agents(self):
        self.send_training_events()

        # Tell agents to act
        for a in self.active_agents:
            if a.available_think_time > 0:
                a.act(self.get_state_for_agent(a))

        # Give agents time to decide
        perm = np.random.permutation(len(self.active_agents))
        self.replay['permutations'].append(perm)
        for i in perm:
            a = self.active_agents[i]
            if a.available_think_time > 0:
                action, think_time = a.wait_for_act()
                self.logger.info(f'Agent <{a.name}> chose action {action} in {think_time:.2f}s.')
                if think_time > a.available_think_time:
                    self.logger.warning(f'Agent <{a.name}> exceeded think time by {s.TIMEOUT - think_time}s. Setting action to "WAIT" and decreasing available time for next round.')
                    action = "WAIT"
                    a.available_think_time = s.TIMEOUT - (think_time - a.available_think_time)
                else:
                    self.logger.info(f'Agent <{a.name}> stayed within acceptable think time.')
                    a.available_think_time = s.TIMEOUT
            else:
                self.logger.info(f'Skipping agent <{a.name}> because of last slow think time.')
                a.available_think_time += s.TIMEOUT
                action = "WAIT"

            self.replay['actions'][a.name].append(action)
            self.perform_agent_action(a, action)

    def end_round(self):
        assert self.running, "End of round requested while not running"
        super().end_round()

        self.logger.info(f'WRAPPING UP ROUND #{self.round}')
        # Clean up survivors
        for a in self.active_agents:
            a.add_event(e.SURVIVED_ROUND)

        # Send final event to agents that expect them
        for a in self.agents:
            if a.train:
                a.round_ended()

        # Save course of the game for future replay
        if self.args.save_replay:
            self.replay['n_steps'] = self.step
            name = f'replays/{self.round_id}.pt' if self.args.save_replay is True else self.args.save_replay
            with open(name, 'wb') as f:
                pickle.dump(self.replay, f)

        # Mark round as ended
        self.running = False

        self.logger.debug('Setting ready_for_restart_flag')
        self.ready_for_restart_flag.set()

    def end(self):
        if self.running:
            self.end_round()
        self.logger.info('SHUT DOWN')
        for a in self.agents:
            # Send exit message to shut down agent
            self.logger.debug(f'Sending exit message to agent <{a.name}>')




class GUI:
    def __init__(self, args: WorldArgs, world: GenericWorld):
        self.args = args
        self.world = world

        # Initialize screen
        self.screen = pygame.display.set_mode((s.WIDTH, s.HEIGHT))
        pygame.display.set_caption('BombeRLe')
        icon = pygame.image.load(f'assets/bomb_yellow.png')
        pygame.display.set_icon(icon)

        # Background and tiles
        self.background = pygame.Surface((s.WIDTH, s.HEIGHT))
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))
        self.t_wall = pygame.image.load('assets/brick.png')
        self.t_crate = pygame.image.load('assets/crate.png')

        # Font for scores and such
        font_name = dirname(__file__) + '/assets/emulogic.ttf'
        self.fonts = {
            'huge': pygame.font.Font(font_name, 20),
            'big': pygame.font.Font(font_name, 16),
            'medium': pygame.font.Font(font_name, 10),
            'small': pygame.font.Font(font_name, 8),
        }

        self.frame = 0

    def render_text(self, text, x, y, color, halign='left', valign='top', size='medium', aa=False):
        text_surface = self.fonts[size].render(text, aa, color)
        text_rect = text_surface.get_rect()
        if halign == 'left':   text_rect.left = x
        if halign == 'center': text_rect.centerx = x
        if halign == 'right':  text_rect.right = x
        if valign == 'top':    text_rect.top = y
        if valign == 'center': text_rect.centery = y
        if valign == 'bottom': text_rect.bottom = y
        self.screen.blit(text_surface, text_rect)

    def render(self):
        self.frame += 1
        self.screen.blit(self.background, (0, 0))

        # World
        for x in range(self.world.arena.shape[1]):
            for y in range(self.world.arena.shape[0]):
                if self.world.arena[x, y] == -1:
                    self.screen.blit(self.t_wall,
                                     (s.GRID_OFFSET[0] + s.GRID_SIZE * x, s.GRID_OFFSET[1] + s.GRID_SIZE * y))
                if self.world.arena[x, y] == 1:
                    self.screen.blit(self.t_crate,
                                     (s.GRID_OFFSET[0] + s.GRID_SIZE * x, s.GRID_OFFSET[1] + s.GRID_SIZE * y))
        self.render_text(f'Step {self.world.step:d}', s.GRID_OFFSET[0], s.HEIGHT - s.GRID_OFFSET[1] / 2, (64, 64, 64),
                         valign='center', halign='left', size='medium')

        # Items
        for bomb in self.world.bombs:
            bomb.render(self.screen, s.GRID_OFFSET[0] + s.GRID_SIZE * bomb.x, s.GRID_OFFSET[1] + s.GRID_SIZE * bomb.y)
        for coin in self.world.coins:
            if coin.collectable:
                coin.render(self.screen, s.GRID_OFFSET[0] + s.GRID_SIZE * coin.x,
                            s.GRID_OFFSET[1] + s.GRID_SIZE * coin.y)

        # Agents
        for agent in self.world.active_agents:
            agent.render(self.screen, s.GRID_OFFSET[0] + s.GRID_SIZE * agent.x,
                         s.GRID_OFFSET[1] + s.GRID_SIZE * agent.y)

        # Explosions
        for explosion in self.world.explosions:
            explosion.render(self.screen)

        # Scores
        # agents = sorted(self.agents, key=lambda a: (a.score, -a.mean_time), reverse=True)
        agents = self.world.agents
        leading = max(agents, key=lambda a: (a.score, a.name))
        y_base = s.GRID_OFFSET[1] + 15
        for i, a in enumerate(agents):
            bounce = 0 if (a is not leading or self.world.running) else np.abs(10 * np.sin(5 * time()))
            a.render(self.screen, 600, y_base + 50 * i - 15 - bounce)
            self.render_text(a.name, 650, y_base + 50 * i,
                             (64, 64, 64) if a.dead else (255, 255, 255),
                             valign='center', size='small')
            for j, trophy in enumerate(a.trophies):
                self.screen.blit(trophy, (660 + 10 * j, y_base + 50 * i + 12))
            self.render_text(f'{a.score:d}', 830, y_base + 50 * i, (255, 255, 255),
                             valign='center', halign='right', size='big')
            self.render_text(f'{a.total_score:d}', 890, y_base + 50 * i, (64, 64, 64),
                             valign='center', halign='right', size='big')

        # End of round info
        if not self.world.running:
            x_center = (s.WIDTH - s.GRID_OFFSET[0] - s.COLS * s.GRID_SIZE) / 2 + s.GRID_OFFSET[0] + s.COLS * s.GRID_SIZE
            color = np.int_((255 * (np.sin(3 * time()) / 3 + .66),
                             255 * (np.sin(4 * time() + np.pi / 3) / 3 + .66),
                             255 * (np.sin(5 * time() - np.pi / 3) / 3 + .66)))
            self.render_text(leading.name, x_center, 320, color,
                             valign='top', halign='center', size='huge')
            self.render_text('has won the round!', x_center, 350, color,
                             valign='top', halign='center', size='big')
            leading_total = max(self.world.agents, key=lambda a: (a.total_score, a.name))
            if leading_total is leading:
                self.render_text(f'{leading_total.name} is also in the lead.', x_center, 390, (128, 128, 128),
                                 valign='top', halign='center', size='medium')
            else:
                self.render_text(f'But {leading_total.name} is in the lead.', x_center, 390, (128, 128, 128),
                                 valign='top', halign='center', size='medium')
