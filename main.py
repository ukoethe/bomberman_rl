import os
import sys
import threading
from argparse import ArgumentParser
from time import sleep, time

import settings as s
from environment import BombeRLeWorld, GenericWorld
from fallbacks import pygame, tqdm, LOADED_PYGAME
from replay import ReplayWorld


# Function to run the game logic in a separate thread
def game_logic(world: GenericWorld, user_inputs, args):
    last_update = time()
    while True:
        now = time()
        if args.turn_based and len(user_inputs) == 0:
            sleep(0.1)
            continue
        elif world.gui is not None and (now - last_update < args.update_interval):
            sleep(args.update_interval - (now - last_update))
            continue

        last_update = now
        if world.running:
            world.do_step(user_inputs.pop(0) if len(user_inputs) else 'WAIT')


def main(argv = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
    play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?', help='Store the game as .pt for a replay')
    play_parser.add_argument("--no-gui", default=False, action="store_true", help="Deactivate the user interface and play as fast as possible.")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--fps", type=int, default=15, help="FPS of the GUI (does not change game)")
        sub.add_argument("--turn-based", default=False, action="store_true",
                         help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1,
                         help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)) + "/logs")

        # Video?
        sub.add_argument("--make-video", default=False, action="store_true",
                         help="Make a video from the game")

    args = parser.parse_args(argv)
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1

    has_gui = not args.no_gui
    if has_gui:
        if not LOADED_PYGAME:
            raise ValueError("pygame could not loaded, cannot run with GUI")
        pygame.init()

    # Initialize environment and agents
    if args.command_name == "play":
        agents = []
        if args.train == 0 and not args.continue_without_training:
            args.continue_without_training = True
        if args.my_agent:
            agents.append((args.my_agent, len(agents) < args.train))
            args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))

        world = BombeRLeWorld(args, agents)
    elif args.command_name == "replay":
        world = ReplayWorld(args)
    else:
        raise ValueError(f"Unknown command {args.command_name}")

    # Emulate Windows process spawning behaviour under Unix (for testing)
    # mp.set_start_method('spawn')

    user_inputs = []

    # Start game logic thread
    t = threading.Thread(target=game_logic, args=(world, user_inputs, args), name="Game Logic")
    t.daemon = True
    t.start()

    # Run one or more games
    for _ in tqdm(range(args.n_rounds)):
        if not world.running:
            world.ready_for_restart_flag.wait()
            world.ready_for_restart_flag.clear()
            world.new_round()

        # First render
        if has_gui:
            world.render()
            pygame.display.flip()

        round_finished = False
        last_frame = time()
        user_inputs.clear()

        # Main game loop
        while not round_finished:
            if has_gui:
                # Grab GUI events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if world.running:
                            world.end_round()
                        world.end()
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in (pygame.K_q, pygame.K_ESCAPE):
                            world.end_round()
                        if not world.running:
                            round_finished = True
                        # Convert keyboard input into actions
                        if s.INPUT_MAP.get(key_pressed):
                            if args.turn_based:
                                user_inputs.clear()
                            user_inputs.append(s.INPUT_MAP.get(key_pressed))

                # Render only once in a while
                if time() - last_frame >= 1 / args.fps:
                    world.render()
                    pygame.display.flip()
                    last_frame = time()
                else:
                    sleep_time = 1 / args.fps - (time() - last_frame)
                    if sleep_time > 0:
                        sleep(sleep_time)
            elif not world.running:
                round_finished = True
            else:
                # Non-gui mode, check for round end in 1ms
                sleep(0.001)

    world.end()


if __name__ == '__main__':
    main()
