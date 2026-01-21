from evaluation_bandit import simulation, algorithms, utils
import argparse
import math
import statistics
import os
import json

args = argparse.ArgumentParser()
args.add_argument(
    "--method",
    type=str,
    required=True,
)
args.add_argument(
    "--seeds",
    type=int,
    default=10,
)
args = args.parse_args()

if args.method == "baseline":
    output = simulation.simulate(
        fn=algorithms.uniform,
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
elif args.method == "baseline_nonsquare":
    output = simulation.simulate(
        fn=algorithms.uniform_random,
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
elif args.method == "successive_rejects_constant":
    output = simulation.simulate(
        fn=algorithms.successive_rejects,
        fn_kwargs=dict(phases="constant"),
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
elif args.method == "stochastic_sampling_ranksmooth":

    def sampling_fn_ranksmooth(ys, rank, total):
        return 1 / (rank + 1)

    output = simulation.simulate(
        fn=algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_ranksmooth),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
elif args.method == "stochastic_sampling_bolzmann":

    def sampling_fn_bolzmann(ys, rank, total, temperature=10):
        return math.exp(statistics.mean(ys) / temperature)

    output = simulation.simulate(
        fn=algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_bolzmann),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
elif args.method == "stochastic_sampling_epsilongreedy":

    def sampling_fn_epsilongreedy(ys, rank, total, epsilon=0.5):
        if rank == 0:
            return 1.0 - epsilon
        else:
            return epsilon / (total - 1)

    output = simulation.simulate(
        fn=algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_epsilongreedy),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
elif args.method == "ambiguity_reduction_11":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=1, weight_pairwise=1),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
elif args.method == "ambiguity_reduction_01":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=0, weight_pairwise=1),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
elif args.method == "ambiguity_reduction_10":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=1, weight_pairwise=0),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_all=utils.load_data_bymetrics,
    )
else:
    raise ValueError(f"Unknown method: {args.method}")


os.makedirs("computed/", exist_ok=True)
with open(f"computed/simulation_wmt_metrics_{args.method}.json", "w") as f:
    json.dump(output, f)


"""
rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/evaluation-bandit
rsync -azP euler:/cluster/work/sachan/vilem/evaluation-bandit/computed/ ./computed/

function sbatch_cpu() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
    -J $JOB_NAME \
    --output=logs/%x.out --error=logs/%x.err \
	--mail-type END \
	--mail-user vilem.zouhar@gmail.com \
        --ntasks-per-node=1 \
        --cpus-per-task=100 \
        --mem-per-cpu=700M \
        --time=0-4 \
        --wrap="$JOB_WRAP";
}

# for method in baseline successive_rejects_constant stochastic_sampling_ranksmooth stochastic_sampling_bolzmann stochastic_sampling_epsilongreedy ambiguity_reduction_11 ambiguity_reduction_01 ambiguity_reduction_10; do
for method in stochastic_sampling_ranksmooth stochastic_sampling_bolzmann stochastic_sampling_epsilongreedy ambiguity_reduction_11 ambiguity_reduction_01 ambiguity_reduction_10; do
    sbatch_cpu "sim_wmt_metrics_$method" "python3 scripts/05b-simulation_wmt_metrics_compute.py --method $method"
done
"""
