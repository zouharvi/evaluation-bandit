# %%

import json
import glob
import numpy as np


def read_computed(fname):
    with open(fname, "r") as f:
        return json.load(f)


outputs_all = [
    {"name": fname.split("/")[-1].removesuffix(".json"), "data": read_computed(fname)}
    for fname in sorted(glob.glob("../computed/04/*.json"))
    if "random" in fname
]


def area_under_curve(outputs, key):
    return f"{np.mean([x[key] for x in outputs]):.3f}"


keys = ["wtau", "evalfocus", "stability"]


for output_i, output in enumerate(outputs_all):
    if "stability" not in output["data"][0]:
        continue
    print(
        f"{output['name']:>50}",
        *(f"{area_under_curve(output['data'], key)}" for key in keys),
    )
