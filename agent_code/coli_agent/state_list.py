states = []
binary = range(2)
for a in binary:
    for b in binary:
        for c in binary:
            for d in binary:
                for e in binary:
                    states.append([a, b, c, d, e])

with open("indexed_state_list.csv", encoding="utf-8", mode="w") as f:
    for i, state in enumerate(states):
        f.write(f"{state}\n")
