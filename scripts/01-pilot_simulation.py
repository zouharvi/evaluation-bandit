import DUKE
from DUKE import simulation

import subset2evaluate.utils

data = subset2evaluate.utils.load_data_wmt("wmt25", "en-cs_CZ", normalize=False)
results_k1 = simulation.simulate(data, lambda: DUKE.sampler.SamplerCloseUniform(data[0]["scores"].keys(), K=1))
results_uniform_k3 = simulation.simulate(data, lambda: DUKE.sampler.SamplerCloseUniform(data[0]["scores"].keys(), K=3))
results_cluster_k3 = simulation.simulate(data, lambda: DUKE.sampler.SamplerCloseCluster(data[0]["scores"].keys(), K=3))

import matplotlib.pyplot as plt

plt.plot(
    [cost for cost, corr in results_k1],
    [corr for cost, corr in results_k1],
    label="K=1",
)
plt.plot(
    [cost for cost, corr in results_uniform_k3],
    [corr for cost, corr in results_uniform_k3],
    label="K=3 uniform",
)
plt.plot(
    [cost for cost, corr in results_cluster_k3],
    [corr for cost, corr in results_cluster_k3],
    label="K=3 cluster",
)
plt.legend()
plt.show()