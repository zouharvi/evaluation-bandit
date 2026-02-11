# %%

import json
import glob
import numpy as np
import collections


def read_computed(fname):
    with open(fname, "r") as f:
        return json.load(f)


outputs_all = [
    {"name": fname.split("/")[-1].removesuffix(".json"), "data": read_computed(fname)}
    for fname in sorted(glob.glob("../computed/04/*.json"))
    if "random" in fname
]


# area under curve table


def area_under_curve(outputs, key):
    data_by_budget = collections.defaultdict(list)
    for output in outputs:
        data_by_budget[output["budget"]].append(output)
    data_by_budget = sorted(data_by_budget.values(), key=lambda d: d[0]["budget"])

    x = np.trapezoid(
        y=[
            np.mean([x[key] for x in xs])
            for xs in data_by_budget
            if xs[0]["budget"] >= 0.1 and xs[0]["budget"] <= 1.0
        ],
        x=[
            xs[0]["budget"]
            for xs in data_by_budget
            if xs[0]["budget"] >= 0.1 and xs[0]["budget"] <= 1.0
        ],
    )
    return f"{x / (1 - 0.1):.3f}"


keys = {
    "wtau": r"Weighted $\tau$",
    "evalfocus": r"Evaluation focus",
}


for output_i, output in enumerate(outputs_all):
    print(
        f"{output['name']:>50}",
        *(f"{area_under_curve(output['data'], key)}" for key in keys.keys()),
    )
