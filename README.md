# bomberman\_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## Goal
Create a model that can play bomberman game by its own, based on different reward system.

## Approach
- Program a model that can input an environment and output an action at every step for the agent to take next environment
- At every step a new state of environment and agents is created by considering actions from previous state
- Based on the reward system the decision of the model should be influenced 

## What we know

- **Restrictions**
	- No movement towards empty tiles (i.e. black)
	- 0.5 secs to decide or else `state = "WAITED"`
- **One episode** of 400 steps
- All rules in **Settings.py**

## Tasks to solve

- Efficiently collect coins as quickly as possible
- Drop bombs without killing itself, and collect coins
- Train on peaceful\_agent and coin\_collector\_agent
- Train on with rule\_based\_agent for highest score

## Framework structure

- Environment in **environment.py** in BombeRLeWorld
- **Every step** 
	- call **act** to take next step decision

