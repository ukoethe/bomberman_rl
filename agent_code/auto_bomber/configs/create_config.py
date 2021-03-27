import json
from pathlib import Path

# EPSILONS = [0.15, 0.25, 0.35]
# DISCOUNTS = [0.35, 0.5, 0.65, 0.85, 0.95]
# LRS = [0.006, 0.0003, 0.00006, 0.00001, 0.000006]
# POLICIES = ['SOFTMAX', 'IANN']
# TEMPERATURES = [0.3, 0.5, 0.7, 0.9]

EPSILONS = [0.25]
DISCOUNTS = [0.4, 0.7, 0.9]
LRS = [0.006, 0.0003, 0.00009, 0.00001]
POLICIES = ['SOFTMAX', 'IANN']
TEMPERATURES = [0.4, 0.7, 0.9]


def main():
    with open(Path('default_hyper_parameters.json')) as f:
        default = json.load(f)

    for policy in POLICIES:
        tmp = default
        if policy == 'SOFTMAX':
            for temp in TEMPERATURES:
                for discount in DISCOUNTS:
                    for lr in LRS:
                        tmp['discount'] = discount
                        tmp['learning_rate'] = lr
                        tmp['policy'] = policy
                        tmp['temperature'] = temp

                        with open(Path(f"./stable/stable_{policy}_temp{temp}_disc{discount}_lr{lr}.json"), 'w') as f:
                            json.dump(tmp, f)
        elif policy == 'IANN':
            for eps in EPSILONS:
                for temp in TEMPERATURES:
                    for discount in DISCOUNTS:
                        for lr in LRS:
                            tmp['epsilon'] = eps
                            tmp['discount'] = discount
                            tmp['learning_rate'] = lr
                            tmp['policy'] = policy
                            tmp['temperature'] = temp

                            with open(Path(f"./stable/stable_{policy}_eps{eps}_temp{temp}_disc{discount}_lr{lr}.json"), 'w') as f:
                                json.dump(tmp, f)


if __name__ == '__main__':
    main()
