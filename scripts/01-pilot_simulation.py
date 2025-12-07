import DUKE
from DUKE import simulation
import matplotlib.pyplot as plt
import subset2evaluate.utils

data = subset2evaluate.utils.load_data_wmt("wmt25", "en-cs_CZ", normalize=False)
models = list(data[0]["scores"].keys())
results_k1 = simulation.simulate(data, lambda: DUKE.sampler.SamplerCloseUniform(models, K=1))
results_uniform_k3 = simulation.simulate(data, lambda: DUKE.sampler.SamplerCloseUniform(models, K=3))
results_cluster_k3 = simulation.simulate(data, lambda: DUKE.sampler.SamplerCloseCluster(models, K=3))


fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharex=True)
axs[0].plot(
    results_k1[0],
    results_k1[1],
    label="K=1",
)
axs[0].plot(
    results_uniform_k3[0],
    results_uniform_k3[1],
    label="K=3 uniform",
)
axs[0].plot(
    results_cluster_k3[0],
    results_cluster_k3[1],
    label="K=3 cluster",
)

axs[1].plot(
    results_k1[0],
    results_k1[2],
    label="K=1",
)
axs[1].plot(
    results_uniform_k3[0],
    results_uniform_k3[2],
    label="K=3 uniform",
)
axs[1].plot(
    results_cluster_k3[0],
    results_cluster_k3[2],
    label="K=3 cluster",
)

axs[0].set_ylabel("Cluster count")
axs[1].set_ylabel("Correlation with true ranking")
axs[0].set_xlabel("Annotation cost")
axs[1].set_xlabel("Annotation cost")
axs[1].legend()

axs[0].spines[['top', 'right']].set_visible(False)
axs[1].spines[['top', 'right']].set_visible(False)

axs[1].set_ylim(0.5, )

plt.tight_layout()
plt.savefig("figures/pilot_simulation.svg")
plt.show()