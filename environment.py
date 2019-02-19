
from time import time
from datetime import datetime
import multiprocessing as mp
import numpy as np
import random
import pygame
import pickle
from pygame.locals import *
from pygame.transform import smoothscale

import logging

from agents import *
from items import *
from settings import s, e


class BombeRLeWorld(object):

    def __init__(self, agents):
        self.setup_logging()
        if s.gui:
            self.setup_gui()

        # Available robot colors
        self.colors = ['blue', 'green', 'yellow', 'pink']
        self.setup_agents(agents)

        # Get the game going
        self.round = 0
        self.running = False
        self.ready_for_restart_flag = mp.Event()
        self.new_round()


    def setup_logging(self):
        self.logger = logging.getLogger('BombeRLeWorld')
        self.logger.setLevel(s.log_game)
        handler = logging.FileHandler('logs/game.log', mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Initializing game world')


    def setup_gui(self):
        # Initialize screen
        self.screen = pygame.display.set_mode((s.width, s.height))
        pygame.display.set_caption('BombeRLe')
        icon = pygame.image.load(f'assets/bomb_yellow.png')
        pygame.display.set_icon(icon)

        # Background and tiles
        self.background = pygame.Surface((s.width, s.height))
        self.background = self.background.convert()
        self.background.fill((0,0,0))
        self.t_wall = pygame.image.load('assets/brick.png')
        self.t_crate = pygame.image.load('assets/crate.png')

        # Font for scores and such
        font_name = 'assets/emulogic.ttf'
        self.fonts = {
            'huge': pygame.font.Font(font_name, 20),
            'big': pygame.font.Font(font_name, 16),
            'medium': pygame.font.Font(font_name, 10),
            'small': pygame.font.Font(font_name, 8),
        }


    def setup_agents(self, agents):
        # Add specified agents and start their subprocesses
        self.agents = []
        for agent_dir, train in agents:
            if list([d for d,t in agents]).count(agent_dir) > 1:
                name = agent_dir + '_' + str(list([a.process.agent_dir for a in self.agents]).count(agent_dir))
            else:
                name = agent_dir
            self.add_agent(agent_dir, name, train=train)


    def new_round(self):
        if self.running:
            self.logger.warn('New round requested while still running')
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
        self.arena = (np.random.rand(s.cols, s.rows) < s.crate_density).astype(int)
        self.arena[:1, :] = -1
        self.arena[-1:,:] = -1
        self.arena[:, :1] = -1
        self.arena[:,-1:] = -1
        for x in range(s.cols):
            for y in range(s.rows):
                if (x+1)*(y+1) % 2 == 1:
                    self.arena[x,y] = -1

        # Starting positions
        self.start_positions = [(1,1), (1,s.rows-2), (s.cols-2,1), (s.cols-2,s.rows-2)]
        random.shuffle(self.start_positions)
        for (x,y) in self.start_positions:
            for (xx,yy) in [(x,y), (x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                if self.arena[xx,yy] == 1:
                    self.arena[xx,yy] = 0

        # Distribute coins evenly
        self.coins = []
        for i in range(3):
            for j in range(3):
                n_crates = (self.arena[1+5*i:6+5*i, 1+5*j:6+5*j] == 1).sum()
                while True:
                    x, y = np.random.randint(1+5*i,6+5*i), np.random.randint(1+5*j,6+5*j)
                    if n_crates == 0 and self.arena[x,y] == 0:
                        self.coins.append(Coin((x,y)))
                        self.coins[-1].collectable = True
                        break
                    elif self.arena[x,y] == 1:
                        self.coins.append(Coin((x,y)))
                        break

        # Reset agents and distribute starting positions
        for agent in self.agents:
            agent.reset(self.round)
            self.active_agents.append(agent)
            agent.x, agent.y = self.start_positions.pop()

        self.replay = {
                'arena': np.array(self.arena),
                'coins': [c.get_state() for c in self.coins],
                'agents': [a.get_state() for a in self.agents],
                'actions': dict([(a.name, []) for a in self.agents]),
                'permutations': []
            }

        self.running = True


    def add_agent(self, agent_dir, name, train=False):
        if len(self.agents) < s.max_agents:
            # Set up a new process to run the agent's code
            pipe_to_world, pipe_to_agent = mp.Pipe()
            ready_flag = mp.Event()
            train_flag = mp.Event()
            if train:
                train_flag.set()
            p = AgentProcess(pipe_to_world, ready_flag, name, agent_dir, train_flag)
            self.logger.info(f'Starting process for agent <{name}>')
            p.start()

            # Create the agent container object
            agent = Agent(p, pipe_to_agent, ready_flag, self.colors.pop(), train_flag)
            self.agents.append(agent)

            # Make sure process setup is finished
            self.logger.debug(f'Waiting for setup of agent <{agent.name}>')
            agent.ready_flag.wait()
            agent.ready_flag.clear()
            self.logger.debug(f'Setup finished for agent <{agent.name}>')


    def get_state_for_agent(self, agent, exit=False):
        state = {}
        state['step'] = self.step
        state['arena'] = np.array(self.arena)
        state['self'] = agent.get_state()
        state['others'] = [other.get_state() for other in self.active_agents if other is not agent]
        state['bombs'] = [bomb.get_state() for bomb in self.bombs]
        state['coins'] = [coin.get_state() for coin in self.coins if coin.collectable]
        explosion_map = np.zeros(self.arena.shape)
        for e in self.explosions:
            for (x,y) in e.blast_coords:
                explosion_map[x,y] = max(explosion_map[x,y], e.timer)
        state['explosions'] = explosion_map
        state['user_input'] = self.user_input
        state['exit'] = exit
        return state


    def tile_is_free(self, x, y):
        is_free = (self.arena[x,y] == 0)
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free


    def perform_agent_action(self, agent, action):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.events.append(e.MOVED_UP)
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.events.append(e.MOVED_DOWN)
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.events.append(e.MOVED_LEFT)
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.events.append(e.MOVED_RIGHT)
        elif action == 'BOMB' and agent.bombs_left > 0:
            self.logger.info(f'Agent <{agent.name}> drops bomb at {(agent.x, agent.y)}')
            self.bombs.append(agent.make_bomb())
            agent.bombs_left -= 1
            agent.events.append(e.BOMB_DROPPED)
        elif action == 'WAIT':
            agent.events.append(e.WAITED)
        else:
            agent.events.append(e.INVALID_ACTION)


    def poll_and_run_agents(self):
        # Send world state to all agents
        for a in self.active_agents:
            self.logger.debug(f'Sending game state to agent <{a.name}>')
            a.pipe.send(self.get_state_for_agent(a))

        # Send events to all agents that expect them, then reset and wait for them
        for a in self.active_agents:
            if a.train_flag.is_set():
                self.logger.debug(f'Sending event queue {a.events} to agent <{a.name}>')
                a.pipe.send(a.events)
            a.events = []
        for a in self.active_agents:
            if a.train_flag.is_set():
                self.logger.debug(f'Waiting for agent <{a.name}> to process events')
                a.ready_flag.wait()
                self.logger.debug(f'Clearing flag for agent <{a.name}>')
                a.ready_flag.clear()

        # Give agents time to decide and set their ready flags; interrupt after time limit
        deadline = time() + s.timeout
        for a in self.active_agents:
            if not a.ready_flag.wait(deadline - time()):
                self.logger.warn(f'Interrupting agent <{a.name}>')
                if os.name == 'posix':
                    if not a.ready_flag.is_set():
                        os.kill(a.process.pid, signal.SIGINT)
                else:
                    # Special case for Windows
                    if not a.ready_flag.is_set():
                        os.kill(a.process.pid, signal.CTRL_C_EVENT)
                a.events.append(e.INTERRUPTED)

        # Perform decided agent actions
        perm = np.random.permutation(len(self.active_agents))
        self.replay['permutations'].append(perm)
        for i in perm:
            a = self.active_agents[i]
            self.logger.debug(f'Collecting action from agent <{a.name}>')
            (action, t) = a.pipe.recv()
            self.logger.info(f'Agent <{a.name}> chose action {action} in {t:.2f}s.')
            a.times.append(t)
            a.mean_time = np.mean(a.times)
            self.replay['actions'][a.name].append(action)

            self.perform_agent_action(a, action)

        # Reset agent flags
        for a in self.active_agents:
            self.logger.debug(f'Clearing flag for agent <{a.name}>')
            a.ready_flag.clear()


    def put_down_agent(self, agent):
        # Send exit message to end round for this agent
        self.logger.debug(f'Send exit message to end round for {agent.name}')
        agent.pipe.send(self.get_state_for_agent(agent, exit=True))
        agent.ready_flag.wait()
        agent.ready_flag.clear()


    def do_step(self, user_input='WAIT'):
        self.step += 1
        self.logger.info(f'STARTING STEP {self.step}')

        self.user_input = user_input
        self.logger.debug(f'User input: {self.user_input}')

        self.poll_and_run_agents()

        # Coins
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        self.logger.info(f'Agent <{a.name}> picked up coin at {(a.x, a.y)} and receives 1 point')
                        a.update_score(s.reward_coin)
                        a.events.append(e.COIN_COLLECTED)
                        a.trophies.append(Agent.coin_trophy)

        # Bombs
        for bomb in self.bombs:
            # Explode when timer is finished
            if bomb.timer <= 0:
                self.logger.info(f'Agent <{bomb.owner.name}>\'s bomb at {(bomb.x, bomb.y)} explodes')
                bomb.owner.events.append(e.BOMB_EXPLODED)
                blast_coords = bomb.get_blast_coords(self.arena)
                # Clear crates
                for (x,y) in blast_coords:
                    if self.arena[x,y] == 1:
                        self.arena[x,y] = 0
                        bomb.owner.events.append(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x,c.y) == (x,y):
                                c.collectable = True
                                self.logger.info(f'Coin found at {(x,y)}')
                                bomb.owner.events.append(e.COIN_FOUND)
                # Create explosion
                screen_coords = [(s.grid_offset[0] + s.grid_size*x, s.grid_offset[1] + s.grid_size*y) for (x,y) in blast_coords]
                self.explosions.append(Explosion(blast_coords, screen_coords, bomb.owner))
                bomb.active = False
                bomb.owner.bombs_left += 1
            # Progress countdown
            else:
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]

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
                            a.events.append(e.KILLED_SELF)
                            explosion.owner.trophies.append(Agent.suicide_trophy)
                        else:
                            self.logger.info(f'Agent <{a.name}> blown up by agent <{explosion.owner.name}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(s.reward_kill)
                            explosion.owner.events.append(e.KILLED_OPPONENT)
                            explosion.owner.trophies.append(smoothscale(a.avatar, (15,15)))
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False
            # Progress countdown
            explosion.timer -= 1
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            a.events.append(e.GOT_KILLED)
            for aa in self.active_agents:
                if aa is not a:
                    aa.events.append(e.OPPONENT_ELIMINATED)
            self.put_down_agent(a)
        self.explosions = [e for e in self.explosions if e.active]

        if self.time_to_stop():
            self.end_round()


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
        if s.stop_if_not_training:
            if not any([a.train_flag.is_set() for a in self.active_agents]):
                self.logger.info('No training agent left alive, wrap up round')
                return True
        if self.step >= s.max_steps:
            self.logger.info('Maximum number of steps reached, wrap up round')
            return True

        return False


    def end_round(self):
        if self.running:
            # Wait in case there is still a game step running
            sleep(s.update_interval)

            self.logger.info(f'WRAPPING UP ROUND #{self.round}')
            # Clean up survivors
            for a in self.active_agents:
                a.events.append(e.SURVIVED_ROUND)
                self.put_down_agent(a)
            # Send final event queue to agents that expect them
            for a in self.agents:
                if a.train_flag.is_set():
                    self.logger.debug(f'Sending final event queue {a.events} to agent <{a.name}>')
                    a.pipe.send(a.events)
                    a.events = []
                    a.ready_flag.wait()
                    a.ready_flag.clear()
            # Penalty for agent who spent most time thinking
            if len(self.agents) > 1:
                self.replay['times'] = [a.mean_time for a in self.agents]
                slowest = max(self.agents, key=lambda a: a.mean_time)
                self.logger.info(f'Agent <{slowest.name}> loses 1 point for being slowest (avg. {slowest.mean_time:.3f}s)')
                slowest.update_score(s.reward_slow)
                slowest.trophies.append(Agent.time_trophy)
            # Save course of the game for future replay
            if s.save_replay:
                self.replay['n_steps'] = self.step
                with open(f'replays/{self.round_id}.pt', 'wb') as f:
                    pickle.dump(self.replay, f)
            # Mark round as ended
            self.running = False
        else:
            self.logger.warn('End-of-round requested while no round was running')

        self.logger.debug('Setting ready_for_restart_flag')
        self.ready_for_restart_flag.set()


    def end(self):
        if self.running:
            self.end_round()
        self.logger.info('SHUT DOWN')
        for a in self.agents:
            # Send exit message to shut down agent
            self.logger.debug(f'Sending exit message to agent <{a.name}>')
            a.pipe.send(None)


    def render_text(self, text, x, y, color, halign='left', valign='top',
                    size='medium', aa=False):
        if not s.gui: return
        text_surface = self.fonts[size].render(text, aa, color)
        text_rect = text_surface.get_rect()
        if halign == 'left':   text_rect.left    = x
        if halign == 'center': text_rect.centerx = x
        if halign == 'right':  text_rect.right   = x
        if valign == 'top':    text_rect.top     = y
        if valign == 'center': text_rect.centery = y
        if valign == 'bottom': text_rect.bottom  = y
        self.screen.blit(text_surface, text_rect)


    def render(self):
        if not s.gui: return
        self.screen.blit(self.background, (0,0))

        # World
        for x in range(self.arena.shape[1]):
            for y in range(self.arena.shape[0]):
                if self.arena[x,y] == -1:
                    self.screen.blit(self.t_wall, (s.grid_offset[0] + s.grid_size*x, s.grid_offset[1] + s.grid_size*y))
                if self.arena[x,y] == 1:
                    self.screen.blit(self.t_crate, (s.grid_offset[0] + s.grid_size*x, s.grid_offset[1] + s.grid_size*y))
        self.render_text(f'Step {self.step:d}', s.grid_offset[0], s.height - s.grid_offset[1]/2, (64,64,64),
                         valign='center', halign='left', size='medium')

        # Items
        for bomb in self.bombs:
            bomb.render(self.screen, s.grid_offset[0] + s.grid_size*bomb.x, s.grid_offset[1] + s.grid_size*bomb.y)
        for coin in self.coins:
            if coin.collectable:
                coin.render(self.screen, s.grid_offset[0] + s.grid_size*coin.x, s.grid_offset[1] + s.grid_size*coin.y)

        # Agents
        for agent in self.active_agents:
            agent.render(self.screen, s.grid_offset[0] + s.grid_size*agent.x, s.grid_offset[1] + s.grid_size*agent.y)

        # Explosions
        for explosion in self.explosions:
            explosion.render(self.screen)

        # Scores
        # agents = sorted(self.agents, key=lambda a: (a.score, -a.mean_time), reverse=True)
        agents = self.agents
        leading = max(self.agents, key=lambda a: (a.score, -a.mean_time))
        y_base = s.grid_offset[1] + 15
        for i, a in enumerate(agents):
            bounce = 0 if (a is not leading or self.running) else np.abs(10*np.sin(5*time()))
            a.render(self.screen, 600, y_base + 50*i - 15 - bounce)
            self.render_text(a.name, 650, y_base + 50*i,
                             (64,64,64) if a.dead else (255,255,255),
                             valign='center', size='small')
            for j, trophy in enumerate(a.trophies):
                self.screen.blit(trophy, (660 + 10*j, y_base + 50*i + 12))
            self.render_text(f'{a.score:d}', 830, y_base + 50*i, (255,255,255),
                             valign='center', halign='right', size='big')
            self.render_text(f'{a.total_score:d}', 890, y_base + 50*i, (64,64,64),
                             valign='center', halign='right', size='big')
            self.render_text(f'({a.mean_time:.3f})', 930, y_base + 50*i, (128,128,128),
                             valign='center', size='small')

        # End of round info
        if not self.running:
            x_center = (s.width - s.grid_offset[0] - s.cols * s.grid_size) / 2 + s.grid_offset[0] + s.cols * s.grid_size
            color = np.int_((255*(np.sin(3*time())/3 + .66),
                             255*(np.sin(4*time()+np.pi/3)/3 + .66),
                             255*(np.sin(5*time()-np.pi/3)/3 + .66)))
            self.render_text(leading.name, x_center, 320, color,
                             valign='top', halign='center', size='huge')
            self.render_text('has won the round!', x_center, 350, color,
                             valign='top', halign='center', size='big')
            leading_total = max(self.agents, key=lambda a: (a.total_score, -a.mean_time))
            if leading_total is leading:
                self.render_text(f'{leading_total.name} is also in the lead.', x_center, 390, (128,128,128),
                                 valign='top', halign='center', size='medium')
            else:
                self.render_text(f'But {leading_total.name} is in the lead.', x_center, 390, (128,128,128),
                                 valign='top', halign='center', size='medium')



class ReplayWorld(BombeRLeWorld):

    def __init__(self, replay_file):
        assert s.gui, 'Replay only makes sense with active GUI.'
        self.setup_logging()
        self.setup_gui()

        self.logger.info(f'Loading replay file "{replay_file}"')
        self.replay_file = replay_file
        with open(f'replays/{replay_file}.pt', 'rb') as f:
            self.replay = pickle.load(f)
        if not 'n_steps' in self.replay:
            self.replay['n_steps'] = s.max_steps

        # Recreate the agents
        self.colors = ['blue', 'green', 'yellow', 'pink']
        self.agents = [ReplayAgent(name, self.colors.pop(), x, y)
            for (x,y,name,b,s) in self.replay['agents']]
        for i,t in enumerate(self.replay['times']):
            self.agents[i].mean_time = t

        # Get the game going
        self.round = 1
        self.ready_for_restart_flag = mp.Event()
        self.new_round()


    def new_round(self):
        self.logger.info('STARTING REPLAY')
        pygame.display.set_caption(f'{self.replay_file}')

        # Bookkeeping
        self.step = 0
        self.bombs = []
        self.explosions = []
        self.running = True
        self.frame = 0

        # Game world and objects
        self.arena = np.array(self.replay['arena'])
        self.coins = [Coin(xy) for xy in self.replay['coins']]
        self.active_agents = [a for a in self.agents]
        for i, agent in enumerate(self.agents):
            agent.reset()
            agent.x, agent.y = self.replay['agents'][i][:2]
            agent.total_score = 0


    def poll_and_run_agents(self):
        # Perform recorded agent actions
        perm = self.replay['permutations'][self.step-1]
        for i in perm:
            a = self.active_agents[i]
            self.logger.debug(f'Repeating action from agent <{a.name}>')
            action = self.replay['actions'][a.name][self.step-1]
            self.logger.info(f'Agent <{a.name}> chose action {action}.')
            self.perform_agent_action(a, action)


    def time_to_stop(self):
        time_to_stop = super().time_to_stop()
        if self.step == self.replay['n_steps']:
            self.logger.info('Replay ends here, wrap up round')
            time_to_stop = True
        return time_to_stop


    def end_round(self):
        if self.running:
            self.running = False
            # Wait in case there is still a game step running
            sleep(s.update_interval)

            self.logger.info(f'WRAPPING UP REPLAY')
            # Penalty for agent who spent most time thinking
            if len(self.agents) > 1:
                slowest = max(self.agents, key=lambda a: a.mean_time)
                self.logger.info(f'Agent <{slowest.name}> loses 1 point for being slowest (avg. {slowest.mean_time:.3f}s)')
                slowest.update_score(s.reward_slow)
                slowest.trophies.append(Agent.time_trophy)
        else:
            self.logger.warn('End-of-round requested while no round was running')

        self.logger.debug('Setting ready_for_restart_flag')
        self.ready_for_restart_flag.set()


    def render(self):
        super().render()

        # Save screenshot
        if s.make_video_from_replay:
            self.logger.debug(f'Saving screenshot for frame {self.frame}')
            pygame.image.save(self.screen, f'screenshots/{self.replay_file}_{self.frame:05d}.png')
        self.frame += 1


    def end(self):
        # Turn screenshots into videos
        if s.make_video_from_replay:
            self.logger.debug(f'Turning screenshots into video files')
            import subprocess, os, glob
            subprocess.call(['ffmpeg', '-y', '-framerate', f'{s.fps}',
                    '-f', 'image2', '-pattern_type', 'glob', '-i', f'screenshots/{self.replay_file}_*.png',
                    '-preset', 'veryslow', '-tune', 'animation', '-crf', '5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    f'screenshots/{self.replay_file}_video.mp4'])
            subprocess.call(['ffmpeg', '-y', '-framerate', f'{s.fps}',
                    '-f', 'image2', '-pattern_type', 'glob', '-i', f'screenshots/{self.replay_file}_*.png',
                    '-threads', '2', '-tile-columns', '2', '-frame-parallel', '0', '-g', '100', '-speed', '1',
                    '-pix_fmt', 'yuv420p', '-qmin', '0', '-qmax', '10', '-crf', '5', '-b:v', '2M', '-c:v', 'libvpx-vp9',
                    f'screenshots/{self.replay_file}_video.webm'])
            for f in glob.glob(f'screenshots/{self.replay_file}_*.png'):
                os.remove(f)


    def put_down_agent(self, agent): pass
