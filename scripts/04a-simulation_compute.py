from evaluation_bandit import simulation, algorithms
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
    "--method-sorter",
    type=str,
    default="random",
)
args.add_argument(
    "--seeds",
    type=int,
    default=100,
)
args.add_argument(
    "--max-workers",
    type=int,
    default=15,
)
args = args.parse_args()

if args.method_sorter == "random":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(method="random")
elif args.method_sorter == "metricvar":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="metric_var", metric="metric"
    )
elif args.method_sorter == "metricavg":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="metric_avg", metric="metric"
    )
elif args.method_sorter == "metriccons":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="metric_cons", metric="metric"
    )
elif args.method_sorter == "diversity_bleu":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="BLEU"
    )
elif args.method_sorter == "diversity_unigram":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="unigram"
    )
elif args.method_sorter == "diversity_lm":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="lm"
    )
elif args.method_sorter == "cometconfidence":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="comet_instant_confidence"
    )
elif args.method_sorter == "sentinel_mqm":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(method="sentinel_src_mqm")
elif args.method_sorter == "precomet_diffdisc":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="precomet_diffdisc_direct"
    )
else:
    raise ValueError(f"Unknown method_sorter: {args.method_sorter}")


def simulate(fn, fn_kwargs={}, **kwargs):
    return simulation.simulate(
        fn=fn,
        fn_kwargs=fn_kwargs,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
        max_workers=args.max_workers,
        **kwargs,
    )


if args.method == "uniform":
    output = simulate(algorithms.uniform)
elif args.method == "uniform_nonsquare":
    output = simulate(algorithms.uniform_nonsquare, accepts_budgets=True)
elif args.method == "successive_rejects_constant":
    output = simulate(algorithms.successive_rejects, fn_kwargs=dict(phases="constant"))
elif args.method == "weighted_sampling_rank":

    def sampling_fn_rank(ys, rank, total):
        return 1 / (rank + 1)

    output = simulate(
        algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_rank),
        accepts_budgets=True,
    )
elif args.method == "weighted_sampling_ranksqrt":

    def sampling_fn_ranksqrt(ys, rank, total):
        return math.sqrt(1 / (rank + 1))

    output = simulate(
        algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_ranksqrt),
        accepts_budgets=True,
    )
elif args.method == "weighted_sampling_bolzmann":

    def sampling_fn_bolzmann(ys, rank, total, temperature=10):
        return math.exp(statistics.mean(ys) / temperature)

    output = simulate(
        algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_bolzmann),
        accepts_budgets=True,
    )
elif args.method == "weighted_sampling_epsilongreedy":

    def sampling_fn_epsilongreedy(ys, rank, total, epsilon=0.5):
        if rank == 0:
            return 1.0 - epsilon
        else:
            return epsilon / (total - 1)

    output = simulate(
        algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_epsilongreedy),
        accepts_budgets=True,
    )
elif args.method == "weighted_sampling_oracle_ranksqrt":

    def sampling_fn_ranksqrt(ys, rank, total):
        return math.sqrt(1 / (rank + 1))

    output = simulate(
        algorithms.weighted_sampling_oracle,
        fn_kwargs=dict(sampling_fn=sampling_fn_ranksqrt),
        accepts_budgets=True,
    )
elif args.method == "weighted_sampling_oracle_rank":

    def sampling_fn_rank(ys, rank, total):
        return 1 / (rank + 1)

    output = simulate(
        algorithms.weighted_sampling_oracle,
        fn_kwargs=dict(sampling_fn=sampling_fn_rank),
        accepts_budgets=True,
    )
elif args.method == "ucb":
    output = simulate(
        algorithms.upper_confidence_bound,
        fn_kwargs=dict(topk=3, c=50),
        accepts_budgets=True,
    )
elif args.method == "ambiguity_reduction_11":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=1, weight_pairwise=1),
        accepts_budgets=True,
    )
elif args.method == "ambiguity_reduction_01":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=0, weight_pairwise=1),
        accepts_budgets=True,
    )
elif args.method == "ambiguity_reduction_10":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=1, weight_pairwise=0),
        accepts_budgets=True,
    )
else:
    raise ValueError(f"Unknown method: {args.method}")


os.makedirs("computed/04/", exist_ok=True)
with open(f"computed/04/{args.method}_{args.method_sorter}.json", "w") as f:
    json.dump(output, f)


"""
rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/evaluation-bandit
rsync -azP euler:/cluster/work/sachan/vilem/evaluation-bandit/computed/04/ ./computed/04/

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
        --mem-per-cpu=600M \
        --time=0-4 \
        --wrap="$JOB_WRAP";
}

function sbatch_gpu_bigmem() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
    -J $JOB_NAME \
    --output=logs/%x.out --error=logs/%x.err \
    --gpus=1 --gres=gpumem:22g \
	--mail-type END \
	--mail-user vilem.zouhar@gmail.com \
        --ntasks-per-node=1 \
        --cpus-per-task=20 \
        --mem-per-cpu=6G \
        --time=0-4 \
        --wrap="$JOB_WRAP";
}

python3 scripts/04a-simulation_compute.py --method uniform_nonsquare --method-sorter random --seeds 100
python3 scripts/04a-simulation_compute.py --method weighted_sampling_rank --method-sorter random --seeds 100
python3 scripts/04a-simulation_compute.py --method weighted_sampling_oracle_rank --method-sorter random --seeds 100
python3 scripts/04a-simulation_compute.py --method uniform --method-sorter random --seeds 100
python3 scripts/04a-simulation_compute.py --method successive_rejects_constant --method-sorter random --seeds 100

for method in uniform uniform_nonsquare successive_rejects_constant weighted_sampling_rank weighted_sampling_oracle_rank weighted_sampling_bolzmann weighted_sampling_epsilongreedy ucb ambiguity_reduction_11 ambiguity_reduction_01 ambiguity_reduction_10; do
for method_sorter in random metricvar metricavg diversity_bleu diversity_unigram; do
    sbatch_cpu "simulation_${method}_${method_sorter}" "python3 scripts/04a-simulation_compute.py --method $method --method-sorter $method_sorter --seeds 100 --max-workers 95";
done
done

for method in uniform uniform_nonsquare successive_rejects_constant weighted_sampling_rank weighted_sampling_oracle_rank weighted_sampling_bolzmann weighted_sampling_epsilongreedy ucb; do
for method_sorter in cometconfidence sentinel_mqm precomet_diffdisc diversity_lm; do
    sbatch_gpu_bigmem "simulation_${method}_${method_sorter}" "python3 scripts/04a-simulation_compute.py --method $method --method-sorter $method_sorter --seeds 1 --max-workers 95";
done
done
"""
