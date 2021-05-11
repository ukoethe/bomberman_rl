import os
import threading
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time

import settings as s
from environment import BombeRLeWorld, GenericWorld, GUI
from fallbacks import pygame, tqdm, LOADED_PYGAME
from replay import ReplayWorld

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


def gui_controller(world, gui, args, user_inputs, stop_flag: threading.Event, quit_flag: threading.Event):
    if args.make_video and not gui.screenshot_dir.exists():
        gui.screenshot_dir.mkdir()

    was_running = False
    while True:
        # Render (which takes time)
        last_frame = time()
        gui.render()
        pygame.display.flip()

        # Save video if passed from running to not running
        is_running = world.running
        if not is_running and was_running and args.make_video:
            gui.make_video()
        was_running = is_running

        # Then sleep until next frame
        sleep_time = 1 / args.fps - (time() - last_frame)
        if sleep_time > 0:
            sleep(sleep_time)

        # Check GUI events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag.set()
                return
            elif event.type == pygame.KEYDOWN:
                key_pressed = event.key
                # End of game: any usual key quits
                if not world.running and world.round >= args.n_rounds and (key_pressed in ESCAPE_KEYS or key_pressed in s.INPUT_MAP):
                    return

                # Stop round with stop keys
                if world.running and key_pressed in ESCAPE_KEYS:
                    stop_flag.set()
                # Convert keyboard input into actions
                elif key_pressed in s.INPUT_MAP:
                    if args.turn_based:
                        user_inputs.clear()
                    user_inputs.append(s.INPUT_MAP[key_pressed])


def gui_blocker(user_inputs, args, world):
    # Start first round
    yield
    last_update = time()

    while True:
        if world.running:
            # Wait for user input
            if args.turn_based:
                while len(user_inputs) == 0:
                    sleep(0.1)
            # Wait for update interval
            else:
                now = time()
                wait_time = args.update_interval - (now - last_update)
                if wait_time > 0:
                    sleep(wait_time)
                last_update = time()
        elif world.round <= args.n_rounds:
            # Next key tells game to continue
            while len(user_inputs) == 0:
                sleep(0.1)
            user_inputs.pop()
        yield


def game_logic(world: GenericWorld, user_inputs, args, stop_flag: threading.Event, quit_flag: threading.Event, blocker=None):
    def wait():
        if blocker is not None:
            next(blocker)

    for _ in tqdm(range(args.n_rounds)):
        wait()
        world.new_round()

        while world.running:
            wait()
            if stop_flag.is_set() or quit_flag.is_set():
                stop_flag.clear()
                break
            world.do_step(user_inputs.pop(0) if len(user_inputs) else 'WAIT')

        if quit_flag.is_set():
            break
    world.end()


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
    play_parser.add_argument("--match-name", help="Give the match a name")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--fps", type=int, default=-1, help="FPS of the GUI (does not change game; default: match update-interval)")
        sub.add_argument("--turn-based", default=False, action="store_true",
                         help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1,
                         help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

        # Video?
        sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?',
                         help="Make a video from the game")

    args = parser.parse_args(argv)
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1
        args.match_name = Path(args.replay).name

    has_gui = not args.no_gui
    if has_gui:
        if not LOADED_PYGAME:
            raise ValueError("pygame could not loaded, cannot run with GUI")
        if args.fps == -1:
            args.fps = 1 / args.update_interval

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

    # Potential communication from GUI
    user_inputs = []
    stop_flag = threading.Event()
    quit_flag = threading.Event()

    # Launch GUI
    if has_gui:
        gui = GUI(world)
        blocker = gui_blocker(user_inputs, args, world)
        t = threading.Thread(target=game_logic, args=(world, user_inputs, args, stop_flag, quit_flag, blocker))
        t.start()
        gui_controller(world, gui, args, user_inputs, stop_flag, quit_flag)
    else:
        game_logic(world, user_inputs, args, stop_flag, quit_flag)


if __name__ == '__main__':
    main()
