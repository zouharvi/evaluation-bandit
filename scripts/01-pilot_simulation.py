import DUKE

import subset2evaluate.utils

data = subset2evaluate.utils.load_data_wmt("wmt25", "en-cs_CZ", normalize=False)
results_k1 = DUKE.simulate(data, lambda: DUKE.sampler.SamplerCloseUniform(data[0]["scores"].keys(), K=1))
results_k2 = DUKE.simulate(data, lambda: DUKE.sampler.SamplerCloseUniform(data[0]["scores"].keys(), K=2))
results_k4 = DUKE.simulate(data, lambda: DUKE.sampler.SamplerCloseUniform(data[0]["scores"].keys(), K=4))

import matplotlib.pyplot as plt

plt.plot(
    [cost for cost, corr in results_k1],
    [corr for cost, corr in results_k1],
    label="K=1",
)
plt.plot(
    [cost for cost, corr in results_k2],
    [corr for cost, corr in results_k2],
    label="K=2",
)
plt.plot(
    [cost for cost, corr in results_k4],
    [corr for cost, corr in results_k4],
    label="K=4",
)
plt.legend()
plt.show()