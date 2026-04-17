# %%


import json
import tqdm
import statistics


def read_computed(method, method_estimator, data):
    with open(
        f"../computed/03/{data}#{method}#{method_estimator}.json",
        "r",
    ) as f:
        return json.load(f)


outputs = [
    {
        "method_typst": "Uniform",
        "method": "uniform",
    },
    {
        "method_typst": "Uniform (sampling)",
        "method": "uniform_nonsquare",
    },
    {
        "method_typst": "Successive rejects",
        "method": "successive_rejects",
    },
    {
        "method_typst": "Sampling rank",
        "method": "weighted_sampling_rank",
    },
    {
        "method_typst": "Sampling $epsilon$-greedy",
        "method": "weighted_sampling_epsilongreedy",
    },
    {
        "method_typst": "Sampling Bolzmann",
        "method": "weighted_sampling_bolzmann",
    },
    {
        "method_typst": "Upper confidence bound",
        "method": "ucb",
    },
    {
        "method_typst": "Confusion minimization",
        "method": "confusion_minimization",
    },
    # {
    #     "method_typst": "Greedy oracle",
    #     "method": "greedy_oracle_invariant_wtau_pow2",
    # },
]

outputs = [
    output | {"method_estimator": method_estimator, "data": data}
    for output in outputs
    for data in ["homo", "hetero", "binary", "likert"]
    for method_estimator in ["additive", "mean"]
]

for output in tqdm.tqdm(outputs):
    try:
        output["outputs"] = read_computed(
            output["method"],
            output["method_estimator"],
            output["data"],
        )
    except FileNotFoundError:
        print(
            f"Warning: computed file for method {output['data']}#{output['method']}#{output['method_estimator']} not found."
        )

outputs_out = []
for output in outputs:
    if "outputs" not in output:
        continue
    wtau = statistics.mean([output["wtau"] for output in output["outputs"]])
    outputs_out.append(
        {
            "method": output["method"],
            "method_typst": output["method_typst"],
            "method_estimator": output["method_estimator"],
            "data": output["data"],
            "wtau": f"{wtau:.3f}",
        }
    )

with open("../figures/simulation_synth.json", "w") as f:
    json.dump(outputs_out, f, indent=2)


# %%

import importlib
import statistics
import evaluation_bandit.utils as utils
import matplotlib.pyplot as plt

importlib.reload(utils)

data = utils.data_humanscores_only(
    utils.load_data_synth(
        seed=0,
        models=50,
        items=500,
        heteroscedastic=False,
        bins=None,
    )
)

models = list(data[0]["scores"].keys())
model_means = [
    statistics.mean(item["scores"][model] for item in data) for model in models
]
model_stds = [
    statistics.stdev(item["scores"][model] for item in data) for model in models
]

plt.hist(model_means, color="black")
plt.show()

plt.scatter(model_means, model_stds, color="black")
plt.xlabel("Mean score")
plt.ylabel("Standard deviation of score")
plt.show()
