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
    "--method-estimator",
    type=str,
    default="mean",
    choices=["mean", "additive", "count"],
)
args.add_argument(
    "--method-estimator-eval",
    type=str,
    default="mean",
    choices=["mean", "additive", "count"],
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
    data_sorter_fn = simulation.subset2evaluate_to_sorter(method="random")
elif args.method_sorter == "metricvar":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_var", metric="metric"
    )
elif args.method_sorter == "metricavg":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_avg", metric="metric"
    )
elif args.method_sorter == "metriccons":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_cons", metric="metric"
    )
elif args.method_sorter == "diversity_bleu":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="BLEU"
    )
elif args.method_sorter == "diversity_unigram":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="unigram"
    )
elif args.method_sorter == "diversity_lm":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="lm"
    )
elif args.method_sorter == "cometconfidence":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="comet_instant_confidence"
    )
elif args.method_sorter == "sentinel_mqm":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(method="sentinel_src_mqm")
elif args.method_sorter == "precomet_diffdisc":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="precomet_diffdisc_direct"
    )
else:
    raise ValueError(f"Unknown method_sorter: {args.method_sorter}")


if args.method_estimator == "mean":
    estimator_fn = simulation.mean
elif args.method_estimator == "additive":
    estimator_fn = simulation.additive
elif args.method_estimator == "count":
    estimator_fn = simulation.count
else:
    raise ValueError(f"Unknown method_estimator: {args.method_estimator}")


if args.method_estimator_eval == "mean":
    estimator_eval_fn = simulation.mean
elif args.method_estimator_eval == "additive":
    estimator_eval_fn = simulation.additive
elif args.method_estimator_eval == "count":
    estimator_eval_fn = simulation.count
else:
    raise ValueError(f"Unknown method_estimator_eval: {args.method_estimator_eval}")


def simulate(fn, kwargs_fn={}, **kwargs):
    return simulation.simulate(
        fn=fn,
        kwargs_fn=kwargs_fn,
        seeds=args.seeds,
        data_sorter_fn=data_sorter_fn,
        max_workers=args.max_workers,
        cache_data_sorter=args.method_sorter != "random",
        estimator_fn=estimator_eval_fn,
        **kwargs,
    )


if args.method == "uniform":
    output = simulate(algorithms.uniform)
elif args.method == "uniform_nonsquare":
    output = simulate(algorithms.uniform_nonsquare, accepts_budgets=True)
elif args.method == "successive_rejects_constant":
    output = simulate(algorithms.successive_rejects, kwargs_fn=dict(phases="constant"))
elif args.method == "weighted_sampling_rank":
    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=lambda ys, rank, total: 1 / (rank + 1),
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_rankpow2":
    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=lambda ys, rank, total: 1 / ((rank + 1) ** 2),
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_ranksqrt":
    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=lambda ys, rank, total: math.sqrt(1 / (rank + 1)),
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_bolzmann":
    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=lambda ys, rank, total, temperature=10: math.exp(
                statistics.mean(ys) / temperature
            ),
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_epsilongreedy":
    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=lambda ys, rank, total, epsilon=0.5: 1 / (rank + 1)
            if rank == 0
            else epsilon / (total - 1),
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_oracle_ranksqrt":
    output = simulate(
        algorithms.weighted_sampling_oracle,
        kwargs_fn=dict(sampling_fn=lambda ys, rank, total: math.sqrt(1 / (rank + 1))),
    )
elif args.method == "weighted_sampling_oracle_rank":
    output = simulate(
        algorithms.weighted_sampling_oracle,
        kwargs_fn=dict(sampling_fn=lambda ys, rank, total: 1 / (rank + 1)),
    )
elif args.method == "ucb":
    output = simulate(
        algorithms.upper_confidence_bound,
        kwargs_fn=dict(topk=3, c=50),
    )
elif args.method == "ambiguity_reduction_11":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        kwargs_fn=dict(weight_pointwise=1, weight_pairwise=1),
    )
elif args.method == "ambiguity_reduction_01":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        kwargs_fn=dict(weight_pointwise=0, weight_pairwise=1),
    )
elif args.method == "ambiguity_reduction_10":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        kwargs_fn=dict(weight_pointwise=1, weight_pairwise=0),
    )
elif args.method == "successive_halving":
    output = simulate(algorithms.successive_halving)
elif args.method == "pvalue_rejects":
    output = simulate(
        algorithms.pvalue_rejects,
    )
elif args.method == "thompson_sampling":
    output = simulate(
        algorithms.thompson_sampling,
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

for method in uniform uniform_nonsquare successive_rejects_constant weighted_sampling_rank weighted_sampling_ranksqrt weighted_sampling_rankpow2 weighted_sampling_oracle_rank weighted_sampling_bolzmann weighted_sampling_epsilongreedy ucb ambiguity_reduction_11 ambiguity_reduction_01 ambiguity_reduction_10 successive_halving pvalue_rejects thompson_sampling; do
for method_sorter in random metricvar metricavg diversity_bleu diversity_unigram; do
    sbatch_cpu "simulation_${method}_${method_sorter}" "python3 scripts/04a-simulation_compute.py --method $method --method-sorter $method_sorter --seeds 100 --max-workers 99";
done
done

for method in uniform uniform_nonsquare successive_rejects_constant weighted_sampling_rank weighted_sampling_oracle_rank weighted_sampling_bolzmann weighted_sampling_epsilongreedy ucb successive_halving pvalue_rejects thompson_sampling; do
for method_sorter in cometconfidence sentinel_mqm precomet_diffdisc diversity_lm; do
    sbatch_gpu_bigmem "simulation_${method}_${method_sorter}" "python3 scripts/04a-simulation_compute.py --method $method --method-sorter $method_sorter --seeds 1 --max-workers 99";
done
done
"""
