# %%

import json
import numpy as np
import collections


def read_computed(fname):
    with open(fname, "r") as f:
        return json.load(f)


outputs_uniform = read_computed("../computed/04/uniform_nonsquare_random.json")
outputs_weighted = read_computed(
    "../computed/04/weighted_sampling_ranksqrt_random.json"
)

# %%

# see the gap between the two on wtau based on budget
import matplotlib.pyplot as plt


def y_perbudget(data, key):
    data_by_budget = collections.defaultdict(list)
    for output in data:
        data_by_budget[output["budget"]].append(output)
    data_by_budget = sorted(data_by_budget.items(), key=lambda d: d[1][0]["budget"])
    return (
        [d[0] for d in data_by_budget],
        [np.mean([y[key] for y in d[1]]) for d in data_by_budget],
    )


plt.plot(
    *y_perbudget(outputs_uniform, "evalfocus"),
)
plt.plot(
    *y_perbudget(outputs_weighted, "evalfocus"),
)
plt.show()

# %%


# see the gap between the two on wtau based on specific dataset (output)


def area_under_curve(outputs, key):
    data_by_budget = collections.defaultdict(list)
    for output in outputs:
        data_by_budget[output["budget"]].append(output)
    data_by_budget = sorted(data_by_budget.values(), key=lambda d: d[0]["budget"])

    x = np.trapezoid(
        y=[
            np.mean([x[key] for x in xs])
            for xs in data_by_budget
            if xs[0]["budget"] >= 0.2 and xs[0]["budget"] <= 1.0
        ],
        x=[
            xs[0]["budget"]
            for xs in data_by_budget
            if xs[0]["budget"] >= 0.2 and xs[0]["budget"] <= 1.0
        ],
    )
    return x / (1 - 0.2)


data_name_all = {tuple(x["data_name"]) for x in outputs_uniform}
data_results = [
    (
        f"{'/'.join(data_name).split('_')[0]:>20}",
        area_under_curve(
            [x for x in outputs_weighted if tuple(x["data_name"]) == data_name],
            "evalfocus",
        )
        - area_under_curve(
            [x for x in outputs_uniform if tuple(x["data_name"]) == data_name],
            "evalfocus",
        ),
    )
    for data_name in data_name_all
]
data_results.sort(key=lambda x: x[1])

for data_name, diff in data_results:
    print(f"{data_name} {diff}")
