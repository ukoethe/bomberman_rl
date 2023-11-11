import numpy as np
import settings as s

class BombermanFeatures:
    NUM_FEATURES = 19

    def __init__(self):
        self.field = None
        self.coins = None
        self.bombs = None
        self.explosion_map = None
        self.agent_position = None
        self.others = None
        self.others_position = None

    def state_to_features(self, game_state):
        self.field = np.array(game_state['field'])
        self.coins = np.array(game_state['coins'])
        self.bombs = game_state['bombs']
        self.explosion_map = np.array(game_state['explosion_map'])
        self.explosion_indices = np.array(np.where(self.explosion_map > 0)).T
        _, _, own_bomb, (x, y) = game_state['self']
        self.agent_position = np.array([x, y])
        self.others = game_state['others']
        self.others_position = np.array([opponent[3] for opponent in self.others])

        features = np.zeros(self.NUM_FEATURES)

        coin_directions, coin_distance = self.find_direction_and_distance(self.coins, [x, y])
        crate_directions, crate_distance = self.find_direction_and_distance(np.where(self.field == 1), [x, y])
        opponent_directions, opponent_distance = self.find_direction_and_distance(self.others_position, [x, y])

        if self.crates_nearby(np.array([x,y])) or self.opponents_nearby(np.array([x,y])):
            bombs_with_newly_placed = self.bombs.copy()
            bombs_with_newly_placed.append(((x,y), 4))
            if not self.could_agent_die(np.array([x,y]), bombs=bombs_with_newly_placed.copy()):
                features[4] = 1
        
        if self.field[x-1, y] != -1 and self.field[x-1, y] != 1 :
            features[5] = self.could_agent_die(np.array([x-1, y]), self.bombs.copy())
            
        if self.field[x+1, y] != -1 and self.field[x+1, y] != 1 :
            features[6] = self.could_agent_die(np.array([x+1, y]), self.bombs.copy())
            
        if self.field[x, y-1] != -1 and self.field[x, y-1] != 1 :
            features[7] = self.could_agent_die(np.array([x, y-1]), self.bombs.copy())

        if self.field[x, y+1] != -1 and self.field[x, y+1] != 1 :
            features[8] = self.could_agent_die(np.array([x, y+1]), self.bombs.copy())

        features[9] = self.could_agent_die(np.array([x,y]), self.bombs.copy())

        features[10:14] = np.array([ int(self.explosion_map[x-1,y]!= 0), int(self.explosion_map[x+1,y]!= 0), int(self.explosion_map[x,y-1]!=0), int(self.explosion_map[x,y+1] != 0)  ]) 

        updated_field = self.field.copy()
        for (bomb_x, bomb_y), countdown in self.bombs:
            updated_field[bomb_x][bomb_y] = -1
        
        for opponent_pos in self.others_position:
            updated_field[opponent_pos[0]][opponent_pos[1]] = -1

        features[14:18] = np.array([ int(updated_field[x-1,y] == 0), int(updated_field[x+1,y] == 0), int(updated_field[x,y-1] == 0), int(updated_field[x,y+1] == 0) ])

        features[18] = int(own_bomb)

        coin_weight = coin_distance
        crate_weight = crate_distance * 4
        opponent_weight = opponent_distance 
        
        if opponent_weight <= coin_weight and opponent_weight <= crate_weight:
            features[0:4] = opponent_directions
        elif coin_weight <= crate_weight:
            features[0:4] = coin_directions
        else:
            features[0:4] = crate_directions

        return features

    def find_direction_and_distance(self, targets, starting_point):
        field = self.field.copy()
        for (bomb_x, bomb_y), countdown in self.bombs:
            field[bomb_x, bomb_y] = -1
        for opponent in self.others_position:
            field[opponent[0], opponent[1]] = -1
        for explosion_index in self.explosion_indices:
            field[explosion_index[0], explosion_index[1]] = -1
        for destination in targets:
            field[destination[0], destination[1]] = 2

        parent_list = np.ones(field.shape[0] * field.shape[1]) * -1
        start = starting_point[1] * field.shape[0] + starting_point[0]
        parent_list[start] = start

        if field[starting_point[0],starting_point[1]] == 2: # If player is already on a target field
            return np.array([0,0,0,0]),0
        
        bfs_queue = np.array([start])
        distance_counter = 0
        
        destination_reached = False
        destination = None

        while not destination_reached and distance_counter < len(bfs_queue):
            current_tile = bfs_queue[distance_counter]
            x = current_tile % field.shape[0]
            y = current_tile // field.shape[0]
            if field[x, y] == 2:
                destination_reached = True
                destination = current_tile
            
            else:
                if current_tile % field.shape[0] != 0 and field[x-1, y]!= -1 and parent_list[current_tile - 1] == -1:
                    bfs_queue = np.append(bfs_queue, current_tile-1)
                    parent_list[current_tile-1] = current_tile

                if current_tile % field.shape[0] != field.shape[0] - 1 and field[x+1,y]!= -1 and parent_list[current_tile + 1] == -1:
                    bfs_queue = np.append(bfs_queue, current_tile+1)
                    parent_list[current_tile+1] = current_tile

                if current_tile >= field.shape[0] and field[x,y-1]!= -1 and parent_list[current_tile - field.shape[0]] == -1:
                    bfs_queue = np.append(bfs_queue,current_tile- field.shape[0])
                    parent_list[current_tile- field.shape[0]] = current_tile
    
                if y < field.shape[0] - 1 and field[x,y+1] != -1 and parent_list[current_tile + field.shape[0]] == -1:
                    bfs_queue = np.append(bfs_queue,current_tile+ field.shape[0])
                    parent_list[current_tile + field.shape[0]] = current_tile

            distance_counter = distance_counter + 1
        
        if destination is not None:
            walk = [destination]
            tile = destination
            
            while tile != start:
                tile = int(parent_list[tile])
                walk.append(tile)

            walk = np.flip(walk)
            path_length = len(walk)
            next_position_x = walk[1] % field.shape[0]
            next_position_y = walk[1] // field.shape[0]

            direction = [int(next_position_x < starting_point[0]), int(next_position_x > starting_point[0]), int(next_position_y < starting_point[1]), int(next_position_y > starting_point[1])] 

            return direction, path_length - 1
        else:
            return np.array([0,0,0,0]), np.inf

    
    def crates_nearby(self, position):
        field = self.field.copy()
        x, y = position

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx != 0 and dy != 0:
                    continue 
                for i in range(1, 4):
                    new_x, new_y = x + dx * i, y + dy * i
                    if field[new_x][new_y] == -1:
                        break
                    if field[new_x, new_y] == 1:
                        return True
        return False

    def opponents_nearby(self, position):
        field = self.field.copy()
        x, y = position

        for opponent in self.others_position:
            field[opponent[0], opponent[1]] = 2

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx != 0 and dy != 0:
                    continue 
                for i in range(1, 4):
                    new_x, new_y = x + dx * i, y + dy * i
                    if field[new_x][new_y] == -1:
                        break
                    if field[new_x, new_y] == 2:
                        return True
        return False

    def nearby_bombs(self, position, num_steps_taken):
            x, y = position
            field = self.field.copy()

            for (bomb_x, bomb_y), countdown in self.bombs:
                field[bomb_x, bomb_y] = 10 + countdown - num_steps_taken

            #we want to find the bomb, that would hit the agent with the smallest cooldown
            #since this is the bomb that put him in danger the soonest
            bomb_detected = False
            smallest_bomb_countdown = 100

            # Check if a bomb is on the field of the agent
            if field[x, y] >= 10:
                bomb_detected = True
                smallest_bomb_countdown = min(smallest_bomb_countdown, field[x,y] - 10)

            # Check for bombs to the right
            if smallest_bomb_countdown != 0:
                for i in range(1,4):
                    if field[x+i][y] == -1:
                        break 

                    if field[x+i,y] >= 10:
                        bomb_detected = True
                        smallest_bomb_countdown = min(smallest_bomb_countdown, field[x+i,y] - 10)
                        
            # Check for bombs to the right
            if smallest_bomb_countdown != 0:
                for i in range(1,4):
                    if field[x-i][y] == -1:
                        break

                    if field[x-i,y] >= 10:
                        bomb_detected = True
                        smallest_bomb_countdown = min(smallest_bomb_countdown, field[x-i,y] - 10)

            #check for bombs below 
            if smallest_bomb_countdown != 0:
                for i in range(1,4):
                    if field[x][y+i] == -1:
                        break

                    if field[x,y+i] >= 10:
                        bomb_detected = True
                        smallest_bomb_countdown = min(smallest_bomb_countdown, field[x,y+i] - 10)

            #check for bombs above
            if smallest_bomb_countdown != 0:
                for i in range(1,4):
                    if field[x][y-i] == -1:
                        break 
                    
                    if field[x,y-i] >= 10:
                        bomb_detected = True
                        smallest_bomb_countdown = min(smallest_bomb_countdown, field[x,y-i] - 10) 

            return bomb_detected, smallest_bomb_countdown


    def could_agent_die(self, starting_point, bombs, starting_distance=0):
        updated_field = self.field.copy()
        cols = self.field.shape[0]
        rows = self.field.shape[1]

        # Treat all bombs like walls
        for (bomb_x, bomb_y), countdown in bombs:
            updated_field[bomb_x, bomb_y] = -1

        # Treat all opponents like walls
        for opponent in self.others_position:
            updated_field[opponent[0],opponent[1]] = -1
            updated_field[opponent[0]-1, opponent[1]] = -1
            updated_field[opponent[0]+1, opponent[1]] = -1
            updated_field[opponent[0], opponent[1]+1] = -1
            updated_field[opponent[0], opponent[1]-1] = -1

        parent = np.ones(cols * rows) * -1
        start = starting_point[1] * cols + starting_point[0]
        parent[start] = start
        
        bfs_queue = np.array([start])
        distance_counter = 0

        #distance from start position to current position
        distance = np.ones(cols * rows) * starting_distance

        while distance_counter < len(bfs_queue):
            current_position = bfs_queue[distance_counter]
            dist = distance[current_position]

            for (bomb_x, bomb_y), countdown in bombs:
                if countdown - distance[current_position] == -1 or countdown - distance[current_position] == 0:
                    for i in range(-3,4):
                        if bomb_x + i >= 0 and bomb_x + i < cols:
                            updated_field[bomb_x + i, bomb_y] = -1

                        if bomb_y + i >= 0 and bomb_y + i < rows:
                            updated_field[bomb_x, bomb_y + i] = -1

                elif countdown - distance[current_position] < -1:
                    for i in range(-3,4):
                        if bomb_x + i >= 0 and bomb_x + i < cols and self.field[bomb_x + i, bomb_y] != -1:
                            updated_field[bomb_x + i, bomb_y] = 0

                        if bomb_y + i >= 0 and bomb_y + i < rows and self.field[bomb_x, bomb_y + i] != -1:
                            updated_field[bomb_x, bomb_y + i] = 0

            x = current_position % cols
            y = current_position // rows
            
            bombs_found, min_cooldown = self.nearby_bombs([x,y], distance[current_position])

            if not bombs_found:
                return False # no danger
            
            if min_cooldown == 0:
                distance_counter = distance_counter + 1
                continue
            
            # to the left
            if current_position % cols != 0 and updated_field[x-1, y] != -1 and updated_field[x-1, y] != 1 and updated_field[x-1, y]!= -1 and parent[current_position-1] == -1 and self.explosion_map[x-1,y] - distance[current_position] <= 0:
                bfs_queue = np.append(bfs_queue, current_position-1)
                parent[current_position-1] = current_position
                distance[current_position-1] = distance[current_position] + 1
            
            # up
            if current_position >= cols and updated_field[x, y-1] != -1 and updated_field[x, y-1]!= 1 and updated_field[x, y-1]!= -1 and parent[current_position-cols] == -1 and self.explosion_map[x,y-1] - distance[current_position] <= 0:
                bfs_queue = np.append(bfs_queue,current_position-cols)
                parent[current_position-cols] = current_position
                distance[current_position-cols] = distance[current_position] + 1
            
            # down
            if y < s.ROWS-1 and updated_field[x,y+1] != -1 and updated_field[x, y+1]!= 1 and updated_field[x,y+1] != -1 and parent[current_position+cols] == -1 and self.explosion_map[x,y+1] - distance[current_position] <= 0:
                bfs_queue = np.append(bfs_queue,current_position+cols)
                parent[current_position+cols] = current_position
                distance[current_position+cols] = distance[current_position] + 1

            # right
            if current_position % cols != cols-1 and updated_field[x+1, y] != -1 and updated_field[x+1, y]!=1 and updated_field[x+1, y]!= -1 and parent[current_position+1] == -1 and self.explosion_map[x+1,y] - distance[current_position] <= 0:
                bfs_queue = np.append(bfs_queue, current_position+1)
                parent[current_position+1] = current_position
                distance[current_position+1] = distance[current_position] + 1

            distance_counter = distance_counter + 1

        return True
