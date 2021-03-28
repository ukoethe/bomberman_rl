import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from sys import stdout
from time import sleep

ROUNDS = 300000
MODEL_ROOT_DIR = "./models/opponents"
CONFIGS_DIR = "./configs"
MAX_PARALLEL = 30


class Scheduler:
    def __init__(self):
        self.processes = [(None, None)] * MAX_PARALLEL
        self.next_free = 0

    def wait_for_free(self):
        while True:
            for index, process in enumerate(self.processes):
                if process[0] is None:
                    self.next_free = index
                    return
                if process[0].poll() is not None:
                    self.next_free = index
                    self.processes[index] = (None, None)
                    return
            sleep(30)

    def execute(self, path: Path):
        if self.next_free is None:
            raise Exception("No free slot")

        current = Path(".")
        p = subprocess.Popen(
            [sys.executable, "./main.py", "play", "--my-agent", "auto_bomber", "--train", "1", "--n-rounds",
             f"{ROUNDS}",
             "--no-gui"],
            env=dict(os.environ, MODEL_DIR=MODEL_ROOT_DIR + path.relative_to(current).__str__(),
                     CONFIG_FILE=path.absolute()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print(f"[{datetime.now(tz=None).strftime('%m/%d/%Y, %H:%M:%S')}] Started: {path.__str__()} - pid: {p.pid}")

        self.processes[self.next_free] = (p, path.stem)
        self.next_free = None

    def terminate(self, name):
        for index, process in enumerate(self.processes):
            if process[1] == name:
                process[0].terminate()
                self.processes[index] = (None, None)

    def wait(self):
        for p, n in self.processes:
            if p is not None:
                p.wait()


if __name__ == '__main__':
    scheduler = Scheduler()
    configs_to_process = Path(CONFIGS_DIR).glob("**/*.json")
    for config in configs_to_process:
        scheduler.wait_for_free()
        scheduler.execute(config)
        stdout.flush()

    scheduler.wait()
