import numpy as np

states = []
binary = range(2)
for a in binary:  # in bomb danger zone?
    for b in binary:  # blocked DOWN?
        for c in binary:  # blocked UP?
            for d in binary:  # blocked RIGHT?
                for e in binary:  # blocked LEFT?
                    for f in binary:  # progressed?
                        for g in range(4):  # direction of nearest coin (or crate)
                            for h in range(
                                3
                            ):  # amount of surrounding crates (none, low, high)
                                for i in binary:  # in opponents bomb area?
                                    states.append([a, b, c, d, e, f, g, h, i])

states = np.array(states)

with open("indexed_state_list.txt", encoding="utf-8", mode="w") as f:
    for i, state in enumerate(states):
        f.write(f"{state}\n")
