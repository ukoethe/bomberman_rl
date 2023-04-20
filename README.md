# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.


# Callbacks file
## this file must contain 2 mandatory functions
`act` is called at every time step of the game:
```
def act(self, game_state):
    ...
    return action
```
where action is one of the following 6 strings
`["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"] `



`setup` is called only once at the beginning of the game, and **not** between multiple rounds of the same game:
```
def setup(self):
    ...
    return None
```

## game_state
is the dict accessible by `act`, it contains the whole observable game state as described here:

```
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
```

| key               | **type** | **shape** | **values**                     | **meaning**                            |
|-------------------|----------|-----------|--------------------------------|----------------------------------------|
| **round**         | int      |           |                                |                                        |
| **step**          | int      |           |                                |                                        |
| **field**         | array    | [17,17]   | -1, 0, 1                       | wall , free , crate                    |
| **self**          | array    | [4]       | [str,int,bool,(int,int)]       | [name, score, bombs_left, (x,y)]       |
| **others**        | array    | [n]       | [[str,int,bool,(int,int)],...] | [[name, score, bombs_left, (x,y)],...] |
| **bombs**         | array    | [n]       | [[int,int],...]                | [[x,y],...]                            |
| **coins**         | array    | [n]       | [[int,int],...]                | [[x,y],...]                            |
| **user_input**    | ?        | ?         | ?                              | ?                                      |
| **explosion_map** | array    | [17,17]   | 0,1                            | safe,explosion                         |


# Wall breaker
## run a simulation
To run a simulation the best approach is to use the command:
```
python main.py play --match-name "wall_breaker_t1" --save-replay --n-rounds 10 --my-agent wall_breaker
```
## agent

The code of any agent is provided in a subfolder of `agent_code`.
Such folder contains a subfolder `logs` , icons and the main script `callbacks.py`.

