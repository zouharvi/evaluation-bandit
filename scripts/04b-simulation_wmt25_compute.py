from translation_bandit import simulation, algorithms
import argparse
import math
import statistics
import os
import json

args = argparse.ArgumentParser()
args.add_argument(
    "--method", type=str, required=True,
)
args.add_argument(
    "--seeds", type=int, default=100,
)
args = args.parse_args()



if args.method == "baseline":
    output = simulation.simulate(
        fn=algorithms.baseline,
        seeds=args.seeds,
    )
elif args.method == "successive_rejects_constant":
    output = simulation.simulate(
        fn=algorithms.successive_rejects,
        fn_kwargs=dict(phases="constant"),
        seeds=args.seeds,
    )
elif args.method == "stochastic_sampling_ranksmooth":
    def sampling_fn_ranksmooth(ys, rank, total):
        return 1 / (rank + 1)

    output = simulation.simulate(
        fn=algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_ranksmooth),
        accepts_budgets=True,
        seeds=args.seeds,
    )
elif args.method == "stochastic_sampling_bolzmann":
    def sampling_fn_bolzmann(ys, rank, total, temperature=1):
        return math.exp(statistics.mean(ys) / temperature)

    output = simulation.simulate(
        fn=algorithms.weighted_sampling,
        fn_kwargs=dict(sampling_fn=sampling_fn_bolzmann),
        accepts_budgets=True,
        seeds=args.seeds,
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
    )
elif args.method == "ambiguity_reduction_11":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=1, weight_pairwise=1),
        accepts_budgets=True,
        seeds=args.seeds,
    )
elif args.method == "ambiguity_reduction_01":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=0, weight_pairwise=1),
        accepts_budgets=True,
        seeds=args.seeds,
    )
elif args.method == "ambiguity_reduction_10":
    output = simulation.simulate(
        fn=algorithms.statistical_ambiguity_reduction,
        fn_kwargs=dict(weight_pointwise=1, weight_pairwise=0),
        accepts_budgets=True,
        seeds=args.seeds,
    )
elif args.method == "s2e_baseline":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="random"),
        seeds=args.seeds,
    )
elif args.method == "s2e_metricvar":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="metric_var", metric="MetricX-25"),
    )
elif args.method == "s2e_metricavg":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="metric_avg", metric="MetricX-25"),
    )
elif args.method == "s2e_metriccons":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="metric_cons", metric="MetricX-25"),
    )
elif args.method == "s2e_diversity_bleu":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="diversity", metric="BLEU"),
    )
elif args.method == "s2e_diversity_unigram":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="diversity", metric="unigram"),
    )
elif args.method == "s2e_diversity_lm":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="diversity", metric="lm"),
        max_workers=1,
    )
elif args.method == "s2e_random":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="random"),
    )
elif args.method == "s2e_cometconfidence":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="comet_instant_confidence"),
        max_workers=1,
    )
elif args.method == "s2e_sentinel_mqm":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="sentinel_src_mqm"),
        max_workers=1,
    )
elif args.method == "s2e_precomet_diffdisc":
    output = simulation.simulate_subset2evaluate(
        fn_kwargs=dict(method="precomet_diffdisc_direct"),
        max_workers=1,
    )
elif args.method == "s2e_diffuse":
    output = simulation.simulate_subset2evaluate_perbudget(
        fn_kwargs=dict(method="diffuse"),
        max_workers=4,
    )
elif args.method == "s2e_kmeans":
    output = simulation.simulate_subset2evaluate_perbudget(
        fn_kwargs=dict(method="kmeans", features="src"),
        max_workers=1,
    )
elif args.method == "s2e_brute_greedy":
    output = simulation.simulate_subset2evaluate_perbudget(
        fn_kwargs=dict(method="bruteforce_greedy", metric="MetricX-25"),
    )
elif args.method == "s2e_brute":
    output = simulation.simulate_subset2evaluate_perbudget(
        fn_kwargs=dict(method="bruteforce", metric="MetricX-25"),
    )
else:
    raise ValueError(f"Unknown method: {args.method}")




os.makedirs("computed/", exist_ok=True)
with open(f"computed/simulation_wmt25_{args.method}.json", "w") as f:
    json.dump(output, f)


"""
rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/translation-bandit
rsync -azP euler:/cluster/work/sachan/vilem/translation-bandit/computed/ ./computed/

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
        --mem-per-cpu=100M \
        --time=0-2 \
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
        --mem-per-cpu=2G \
        --time=0-2 \
        --wrap="$JOB_WRAP";
}

for method in baseline successive_rejects_constant stochastic_sampling_ranksmooth stochastic_sampling_bolzmann stochastic_sampling_epsilongreedy ambiguity_reduction_11 ambiguity_reduction_01 ambiguity_reduction_10; do
    sbatch_cpu "sim_wmt25_$method" "python3 scripts/04b-simulation_wmt25_compute.py --method $method"
done

for method in s2e_metricvar s2e_metricavg s2e_metriccons s2e_diversity_bleu s2e_diversity_unigram; do
    sbatch_cpu "sim_wmt25_$method" "python3 scripts/04b-simulation_wmt25_compute.py --method $method"
done

for method in s2e_diversity_lm s2e_cometconfidence  s2e_sentinel_mqm s2e_precomet_diffdisc; do
    sbatch_gpu "sim_wmt25_$method" "python3 scripts/04b-simulation_wmt25_compute.py --method $method"
done

for method in s2e_kmeans s2e_diffuse s2e_brute_greedy s2e_brute; do
    sbatch_gpu "sim_wmt25_$method" "python3 scripts/04b-simulation_wmt25_compute.py --method $method"
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
        --mem-per-cpu=2G \
        --time=0-2 \
        --wrap="$JOB_WRAP";
}
method=s2e_cometconfidence
sbatch_gpu_bigmem "sim_wmt25_$method" "python3 scripts/04b-simulation_wmt25_compute.py --method $method"

"""