from collections import deque
from random import shuffle
from queue import PriorityQueue
import numpy as np

import settings as s

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def isValid(grid,node):
    if(node[0]<0 or node[1]<0 or node[0]>=grid.shape[0] or node[1]>=grid.shape[0]): return False
    if(grid[node[1],node[0]]==0):return False
    else: return True

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def neighbors(node, grid):
    neigh=[]
    (x,y)=node
    for i in [-1, 1]:
        node1= (x+i,y)
        node2= (x,y+i)
        if isValid(grid,node1): neigh.append(node1)
        if isValid(grid,node2): neigh.append(node2)
    return neigh

def reconstruct_path(came_from, start, goal):
    current= goal
    path= []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    #path.append(start) # optional
    path.reverse() # optional
    return path

def a_star_search(grid, start, goal, self):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from= {}
    cost_so_far= {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current= frontier.get()
        #Non funziona qualcosa
        if current == goal:
            break
        neigh=neighbors(current, grid)
        self.logger.debug(f'Neighbours of {current}: {neigh}') 
        for next in neigh:
            new_cost = cost_so_far[current] + 1#grid[y_node,x_node]
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                self.logger.debug(f'cost until {next} {cost_so_far[next]}') 
                h=heuristic(next, goal)
                priority = new_cost + h
                self.logger.debug(f'heuristic of {next}--{goal}: h{h}') 
                frontier.put(next, priority)
                self.logger.debug(f'priority of {next} {priority}')
                came_from[next] = current
    return came_from, cost_so_far

def print_path_cost(path,grid):
    costs=[]
    for node in path:
        (x,y)=node
        costs.append(grid[y,x])        
    return costs

def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.debug(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~') 
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    #Da tenere??
    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all (????)
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    #Survival agent
    #Detect enemies
    enemies=others
    arena_dim=arena.shape[0]
    free_space = arena == 0

    state_value_matrix = np.matrix(np.ones((arena_dim,arena_dim)) * np.inf)
    #fill manhattan distance matrix from enemies
    
    for o in others:
        temp_value_matrix=np.zeros((arena.shape[0], arena.shape[0]))
        for i in range(0, arena.shape[0]):
            for j in range(0, arena.shape[0]):
                temp_value_matrix[i,j]=temp_value_matrix[i,j]+ abs(i - o[1]) + abs(j - o[0])
        state_value_matrix=np.minimum(state_value_matrix,temp_value_matrix)
    #put 0 where we have walls and crafts (non legal position)
    free_matrix=np.absolute(np.absolute(arena)-np.ones(arena.shape[0])) 
    state_value_matrix=np.multiply(state_value_matrix, free_matrix)
    #add 1 for each legal movement executable from cell i,j
    for i in range(0, arena.shape[0]-1):
            for j in range(0, arena.shape[0]-1):
                leg_move=0
                if(state_value_matrix[i,j]!=0):
                    if(state_value_matrix[i-1,j]!=0): leg_move+=1
                    if(state_value_matrix[i+1,j]!=0): leg_move+=1
                    if(state_value_matrix[i,j-1]!=0): leg_move+=1
                    if(state_value_matrix[i,j+1]!=0): leg_move+=1
                state_value_matrix[i,j]+=leg_move
    #put 1 in bomb range and 0 in bomb location
    for (xb, yb), t in bombs:
        state_value_matrix[yb,xb]=0
        for i in range(1,4):
            #Possiamo usare isValid
            if isValid(state_value_matrix,(xb-i,yb)):
                state_value_matrix[yb,xb-i]=0
            if isValid(state_value_matrix,(xb+i,yb)):
                state_value_matrix[yb,xb+i]=0
            if isValid(state_value_matrix,(xb,yb-i)):
                state_value_matrix[yb-i,xb]=0
            if isValid(state_value_matrix,node=(xb,yb+i)):
                state_value_matrix[yb+i,xb]=0
    

    # Choose direction with greedy (?)
    best_value=state_value_matrix[y,x]
    x_max, y_max=x,y
    best_action='WAIT'
    if(state_value_matrix[y,x+1]>best_value):
        x_max, y_max=x+1,y
        best_value=state_value_matrix[y,x+1]
        best_action='RIGHT'
    if(state_value_matrix[y,x-1]>best_value):
        x_max, y_max=x-1,y
        best_value=state_value_matrix[y,x-1]
        best_action='LEFT'
    if(state_value_matrix[y-1,x]>best_value):
        x_max, y_max=x,y-1
        best_value=state_value_matrix[y-1,x]
        best_action='DOWN'
    if(state_value_matrix[y+1,x]>best_value):
        x_max, y_max=x,y+1
        best_value=state_value_matrix[y+1,x]
        best_action='UP'
    self.logger.debug(f'State_matrix: \n{state_value_matrix}')   
    self.logger.debug(f'Agent location ({(x,y)})with value {state_value_matrix[y,x]}') 
    
    #self.logger.debug(f'Best_action: \n{best_action}')   
    # return best_action

    # Choose direction as min path to safer cell with a star(?)
    
    goals = np.where(state_value_matrix == np.amax(state_value_matrix))
    listOfGoals = list(zip(goals[0], goals[1]))
    dis=50
    #Take the goal nearest to the agent
    for goal in listOfGoals:
        tempdis=heuristic((goal[1],goal[0]), (x,y))
        if (dis>tempdis):
            dis=tempdis
            true_goal=goal
    (x_goal,y_goal)=true_goal
    true_goal=(y_goal,x_goal)
    self.logger.debug(f'GOAL ({true_goal})') 
    came_from, cost_so_far=a_star_search(state_value_matrix, start=(x,y), goal=(true_goal), self=self)

    path=reconstruct_path(came_from, start=(x,y), goal=true_goal)
     
    self.logger.debug(f'best path to ({true_goal}): \n{path}') 
    self.logger.debug(f'path cost: \n{print_path_cost(path,state_value_matrix)}') 
    if (state_value_matrix[y,x]==0 and not path):
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                # Keep track of chosen action for cycle detection
                if a != 'BOMB':
                    self.logger.debug(f'ON a BOMB! Random action: {a}') 
                    return a
    if not path: best_action='WAIT'
    else:
        if(path[0]==(x+1,y)):best_action= 'RIGHT'
        if(path[0]==(x-1,y)):best_action= 'LEFT'
        if(path[0]==(x,y+1)):best_action= 'DOWN'
        if(path[0]==(x,y-1)):best_action= 'UP'
    self.logger.debug(f'Best_action: {best_action}') 
    return best_action