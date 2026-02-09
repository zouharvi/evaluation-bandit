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
args = args.parse_args()

if args.method_sorter == "random":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(method="random")
elif args.method_sorter == "metricvar":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="metric_var", metric="MetricX-25"
    )
elif args.method_sorter == "metricavg":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="metric_avg", metric="MetricX-25"
    )
elif args.method_sorter == "metriccons":
    fn_data_sorter = simulation.subset2evaluate_to_sorter(
        method="metric_cons", metric="MetricX-25"
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

if args.method == "uniform":
    output = simulation.simulate(
        fn=algorithms.uniform,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
    )
elif args.method == "uniform_nonsquare":
    output = simulation.simulate(
        fn=algorithms.uniform_random,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
    )
elif args.method == "successive_rejects_constant":
    output = simulation.simulate(
        fn=algorithms.successive_rejects,
        fn_kwargs=dict(phases="constant"),
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
    )
elif args.method == "stochastic_sampling_ranksmooth":

    def sampling_fn_ranksmooth(ys, rank, total):
        return 1 / (rank + 1)

    output = simulation.simulate(
        fn=algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_ranksmooth),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
    )
elif args.method == "stochastic_sampling_bolzmann":

    def sampling_fn_bolzmann(ys, rank, total, temperature=10):
        return math.exp(statistics.mean(ys) / temperature)

    output = simulation.simulate(
        fn=algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_bolzmann),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
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
        fn_data_sorter=fn_data_sorter,
    )
elif args.method == "ucb":
    output = simulation.simulate(
        fn=algorithms.upper_confidence_bound,
        fn_kwargs=dict(topk=3, c=50),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
    )
elif args.method == "ambiguity_reduction_11":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=1, weight_pairwise=1),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
    )
elif args.method == "ambiguity_reduction_01":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=0, weight_pairwise=1),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
    )
elif args.method == "ambiguity_reduction_10":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=1, weight_pairwise=0),
        accepts_budgets=True,
        seeds=args.seeds,
        fn_data_sorter=fn_data_sorter,
    )
else:
    raise ValueError(f"Unknown method: {args.method}")


os.makedirs("computed/04/", exist_ok=True)
with open(f"computed/04/{args.method}_{args.method_sorter}.json", "w") as f:
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
        --mem-per-cpu=600M \
        --time=0-4 \
        --wrap="$JOB_WRAP";
}

function sbatch_gpu() {
    JOB_NAME=$1;
    JOB_WRAP=$2;
    mkdir -p logs

    sbatch \
    -J $JOB_NAME \
    --output=logs/%x.out --error=logs/%x.err \
    --gpus=1 --gres=gpumem:12g \
	--mail-type END \
	--mail-user vilem.zouhar@gmail.com \
        --ntasks-per-node=1 \
        --cpus-per-task=10 \
        --mem-per-cpu=6G \
        --time=0-4 \
        --wrap="$JOB_WRAP";
}


for method in baseline successive_rejects_constant stochastic_sampling_ranksmooth stochastic_sampling_bolzmann stochastic_sampling_epsilongreedy ambiguity_reduction_11 ambiguity_reduction_01 ambiguity_reduction_10; do
    sbatch_cpu "sim_wmt_$method" "python3 scripts/04a-simulation_compute.py --method $method"
done

for method in s2e_metricvar s2e_metricavg s2e_metriccons s2e_diversity_bleu s2e_diversity_unigram; do
    sbatch_cpu "sim_wmt_$method" "python3 scripts/04a-simulation_compute.py --method $method"
done

for method in s2e_diversity_lm  s2e_sentinel_mqm s2e_precomet_diffdisc; do
    sbatch_gpu "sim_wmt_$method" "python3 scripts/04a-simulation_compute.py --method $method"
done

for method in s2e_kmeans s2e_diffuse s2e_brute_greedy s2e_brute; do
    sbatch_gpu "sim_wmt_$method" "python3 scripts/04a-simulation_compute.py --method $method"
done


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
method=s2e_diffuse
sbatch_gpu_bigmem "sim_wmt_$method" "python3 scripts/04a-simulation_compute.py --method $method"

method=s2e_cometconfidence
sbatch_gpu_bigmem "sim_wmt_$method" "python3 scripts/04a-simulation_compute.py --method $method"

python3 scripts/04a-simulation_compute.py --method uniform --method-sorter random --seeds 10
python3 scripts/04a-simulation_compute.py --method uniform --method-sorter metricvar --seeds 1
"""
