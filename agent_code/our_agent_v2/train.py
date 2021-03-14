import os
import pickle
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-GUI backend, needed for plotting in non-main thread.
import matplotlib.pyplot as plt

import settings as s
import events as e
from .callbacks import state_to_features, fname, FILENAME

# Transition tuple. (s, a, r, s')
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# ------------------------ HYPER-PARAMETERS -----------------------------------
# General hyper-parameters:
TRANSITION_HISTORY_SIZE = 5000 # Keep only ... last transitions
BATCH_SIZE              = 3000 # Size of batch in TD-learning.
TRAIN_FREQ              = 10   # Train model every ... game.

# Epsilon-Greedy:
EXPLORATION_INIT  = 1
EXPLORATION_MIN   = 0.2
EXPLORATION_DECAY = 0.9995

# Softmax:
TAU_INIT  = 10
TAU_MIN   = 0
TAU_DECAY = 0.999

# N-step TD Q-learning:
GAMMA   = 0.90 # Discount factor.
N_STEPS = 1    # Number of steps to consider real, observed rewards.

# Auxilary:
PLOT_FREQ = 100
# -----------------------------------------------------------------------------

# File name for storage of historical record.
FNAME_DATA = f"{FILENAME}_data.pt"

# Custom events:
SURVIVED_STEP = "SURVIVED_STEP"
DIED_DIRECT_NEXT_TO_BOMB = "DIED_DIRECT_NEXT_TO_BOMB"
ALREADY_KNOW_FIELD = "ALREADY_KNOW_FIELD"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
AWAY_FROM_COIN = "AWAY_FROM_COIN"
BACK_AND_FORTH = "BACK_AND_FORTH"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Ques to store the transition tuples and coordinate history of agent.
    self.transitions        = deque(maxlen=TRANSITION_HISTORY_SIZE) # long term memory of complete step
    self.coordinate_history = deque([], 10)                         # short term memory of agent position
    
    # Set inital epsilon/tau.
    if self.act_strategy == 'eps-greedy':
        self.epsilon = EXPLORATION_INIT
    elif self.act_strategy == 'softmax':
        self.tau = TAU_INIT
    else:
        raise ValueError(f"Unknown act_strategy {self.act_strategy}")

    # For evaluation of the training progress:
    # Check if historic data file exists, load it in or start from scratch.
    if os.path.isfile(FNAME_DATA):
        # Load historical training data.
        with open(FNAME_DATA, "rb") as file:
            self.historic_data = pickle.load(file)
        self.game_nr = max(self.historic_data['games']) + 1
    else:    
        # Start a new historic record.
        self.historic_data = {
            'score' : [],       # subplot 1
            'coins' : [],       # subplot 2
            'exploration' : [], # subplot 3
            'games' : []        # subplot 1,2,3 x-axis
        }
        self.game_nr = 0

    # Initialization
    self.score_in_round  = 0
    self.collected_coins = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    ################# (1) Add own events to hand out rewards #################
                
    if old_game_state:
        _, score, bombs_left, (x, y) = old_game_state['self']
        closest_coin_info_old = closets_coin_distance(old_game_state)

        # penalty on loops:
        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 1:
            events.append(ALREADY_KNOW_FIELD)
            if self.coordinate_history.count((x, y)) > 2:
                events.append(BACK_AND_FORTH)

        self.coordinate_history.append((x, y))

        # penalty on going away from coin vs reward for going closer: 
        if new_game_state:
            closest_coin_info_new = closets_coin_distance(new_game_state)

            if closest_coin_info_new is not None:
                if (closest_coin_info_old - closest_coin_info_new) < 0:
                    events.append(CLOSER_TO_COIN)
                else:
                    events.append(AWAY_FROM_COIN)

        if 'GOT_KILLED' in events:
            # closer to bomb gives higher penalty:
            bombs = old_game_state['bombs']
            bomb_xys = [xy for (xy, t) in bombs]

            step_distance_bombs = []
            for bomb in bombs:
                x_rel_bomb = bomb[0][0] - x
                y_rel_bomb = bomb[0][1] - y
                step_distance_bombs.append(np.abs(x_rel_bomb) +  np.abs(y_rel_bomb))
            step_distance_closest_bomb = sorted(step_distance_bombs)[0]

            if step_distance_closest_bomb < 2:
                events.append(DIED_DIRECT_NEXT_TO_BOMB)

    # reward for surviving:          
    if not 'GOT_KILLED' in events:
        events.append(SURVIVED_STEP)         
    
    ################## (2) Store Transision: #################
    
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    ################# (3) For evaluation purposes: #################
    
    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1

    self.score_in_round += reward_from_events(self, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # ---------- (1) Store last transition tuple: ----------
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # ---------- (2) Decrease the exploration rate: ----------
    if self.act_strategy == 'eps-greedy':    
        if self.epsilon > EXPLORATION_MIN:
            self.epsilon *= EXPLORATION_DECAY
    elif self.act_strategy == 'softmax':
        if self.tau > TAU_MIN:
            self.tau *= TAU_DECAY
    else:
        raise ValueError(f"Unknown act_strategy {self.act_strategy}")

    # ---------- (3) TD Q-learning with batch: ----------
    if len(self.transitions) > BATCH_SIZE and self.game_nr % TRAIN_FREQ == 0:

        batch = random.sample(self.transitions, BATCH_SIZE)
        X, targets = [], []
        for state, action, state_next, reward in batch:
            
            q_update = reward
            
            if state is not None:
                
                if not self.is_initialized: 
                    q_values = np.zeros(self.action_size).reshape(1, -1)
                else:
                    q_values = self.model.predict(state)
                    
                if state_next is not None:
                    if not self.is_initialized:
                        q_update = reward
                    else:
                        maximal_response = np.max(self.model.predict(state_next)) 
                        #old_value = q_values[0][self.actions.index(action)]
                        q_update =  (reward + GAMMA *  maximal_response)           

                q_values[0][self.actions.index(action)] = q_update

                X.append(state[0])
                targets.append(q_values[0])

        #self.model.partial_fit(X, targets)
        self.model.fit(X, targets)
        self.is_initialized = True

    # ---------- (4) Store learned model: ----------
    with open(fname, "wb") as file:
        pickle.dump(self.model, file)
   
    # ---------- (5) Performance evaluation: ----------
    # Total score in this game.
    score   = np.sum(self.score_in_round)
   
    # Append results to each specific list.
    self.historic_data['score'].append(score)
    self.historic_data['coins'].append(self.collected_coins)
    self.historic_data['games'].append(self.game_nr)   
    if self.act_strategy == 'eps-greedy':
        self.historic_data['exploration'].append(self.epsilon)
    elif self.act_strategy == 'softmax':
        self.historic_data['exploration'].append(self.tau)

    # Store the historic record.
    with open(FNAME_DATA, "wb") as file:
        pickle.dump(self.historic_data, file)
    
    # Reset game score, coins collected and one up the game count.
    self.score_in_round  = 0
    self.collected_coins = 0
    self.game_nr += 1
    
    # Plot training progress every n:th game.
    if self.game_nr % PLOT_FREQ == 0:

        # Incorporate the full training history.
        games_list = self.historic_data['games']
        score_list = self.historic_data['score']
        coins_list = self.historic_data['coins']
        explr_list = self.historic_data['exploration']

        # Plotting
        fig, ax = plt.subplots(3, sharex=True)

        # Total score per game.
        ax[0].plot(games_list, score_list)
        ax[0].set_title('Total score per game')
        ax[0].set_ylabel('Score')

        # Collected coins per game.
        ax[1].plot(games_list, coins_list)
        ax[1].set_title('Collected coins per game')
        ax[1].set_ylabel('Coins')

        # Exploration rate (epsilon/tau) per game.
        ax[2].plot(games_list, explr_list)
        if self.act_strategy == 'eps-greedy':        
            ax[2].set_title('$\epsilon$-greedy: Exploration rate $\epsilon$')
            ax[2].set_ylabel('$\epsilon$')
        elif self.act_strategy == 'softmax':
            ax[2].set_title('Softmax: Exploration rate $\\tau$')
            ax[2].set_ylabel('$\\tau$')
        ax[2].set_xlabel('Game #')

        # Export the figure.
        fig.tight_layout()
        plt.savefig(f'TrainEval_{FILENAME}.pdf') 
       

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    survive_step = 0.15
    game_rewards = {
        # my Events:
        SURVIVED_STEP:  survive_step,
        DIED_DIRECT_NEXT_TO_BOMB: -2*survive_step,
        ALREADY_KNOW_FIELD: -0.1,
        CLOSER_TO_COIN: 0.2,
        AWAY_FROM_COIN: -0.25,
        BACK_AND_FORTH: -0.5,
        
        # AWAY_FROM_COIN + ALREADY_KNOW_FIELD + survive_step = - (CLOSER_TO_COIN + survive_step) + survive_step
        # 2*AWAY_FROM_COIN + ALREADY_KNOW_FIELD + survive_step = - (CLOSER_TO_COIN + survive_step) + survive_step
        
        e.MOVED_LEFT: 0,
        e.MOVED_RIGHT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: 0,  # could punish for waiting
        e.INVALID_ACTION: -survive_step,
        
        e.BOMB_DROPPED: -0.1,
        e.BOMB_EXPLODED: 0,

        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 1,
        e.COIN_COLLECTED: 20,

        e.KILLED_OPPONENT: 0,
        e.KILLED_SELF: -6* survive_step,   # maybe include later that distance to bomb is included in penatly 

        e.GOT_KILLED: 0,
        e.OPPONENT_ELIMINATED: 0,
        e.SURVIVED_ROUND: survive_step,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def closets_coin_distance(game_state: dict) -> int:
    '''
    return the relative total distance from the 
    agent to the closest Coin to check where Agent got closer to Coin.
    '''
    
    _, score, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']
    coins_dis = []
    for coin in coins:
        total_step_distance = abs(coin[0]-x) + abs(coin[1]-y)
        coin_dis = (total_step_distance)
        coins_dis.append(coin_dis)
    if coins:
        closest_coin_dis = sorted(coins_dis)[0]
        return closest_coin_dis
    else:
        return None

     