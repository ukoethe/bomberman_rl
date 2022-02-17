# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.


## Debugging your code

In GUI mode, the game logic and the agents' code run in a separate threads.
Some IDEs (like Spyder) do not support setting breakpoints in background threads through the IDE interface. 

To identify bugs using step debugging, you can:

- use a `breakpoint()` statement in the code,
- use a different IDE (like PyCharm, VSCode) that allows setting breakpoints in background threads,
- use the `--no-gui` mode, which executes the agent logic on the main thread.

Stack traces of errors are printed to the log file of each agent.
