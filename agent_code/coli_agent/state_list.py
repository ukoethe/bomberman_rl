import numpy as np

states = []
binary = range(2)
for a in binary:
    for b in binary:
        for c in binary:
            for d in binary:
                for e in binary:
                    for f in binary:
                        for g in range(4):
                            for h in range(3):
                                for i in binary:
                                    states.append([a, b, c, d, e, f, g, h, i])

states = np.array(states)

with open("indexed_state_list.txt", encoding="utf-8", mode="w") as f:
    for i, state in enumerate(states):
        f.write(f"{state}\n")
