# push code to cluster
rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/evaluation-bandit
# get results from cluster
rsync -azP euler:/cluster/work/sachan/vilem/evaluation-bandit/computed/02/ ./computed/02/

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



# random metricvar metricavg diversity_bleu diversity_unigram; do

for method in uniform uniform_nonsquare greedy_oracle greedy_oracle_invariant confusion_minimization successive_rejects_constant weighted_sampling_rank weighted_sampling_rankpow2 weighted_sampling_bolzmann weighted_sampling_epsilongreedy ucb; do
# TODO: run and then get main simulation results(03a-) but also distribution (05a-) based on stored model_estimates
for method_estimator in mean additive; do
for method_sorter in random metricavg metricavg_cost rev_metricavg; do
    sbatch_cpu \
        "simulation_${method}#${method_sorter}#${method_estimator}#${method_estimator}" \
        "python3 scripts/02a-simulation_compute.py --method $method --method-sorter $method_sorter --method-estimator $method_estimator --method-estimator-eval $method_estimator --seeds 100 --max-workers 100" \
    ;
done
done
done


# secondary appendix results
method_estimator=mean
method_sorter=random
for method in ambiguity_reduction_11 ambiguity_reduction_01 ambiguity_reduction_10 successive_halving pvalue_rejects thompson_sampling weighted_sampling_oracle_rank weighted_sampling_oracle_rankpow2; do
    sbatch_cpu \
        "simulation_${method}#${method_sorter}#${method_estimator}" \
        "python3 scripts/02a-simulation_compute.py --method $method --method-sorter $method_sorter --method-estimator $method_estimator --method-estimator-eval $method_estimator --seeds 100 --max-workers 99" \
    ;
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

for method in uniform uniform_nonsquare successive_rejects_constant weighted_sampling_rank weighted_sampling_oracle_rank weighted_sampling_bolzmann weighted_sampling_epsilongreedy ucb successive_halving pvalue_rejects thompson_sampling; do
for method_sorter in cometconfidence sentinel_mqm precomet_diffdisc diversity_lm; do
    sbatch_gpu_bigmem "simulation_${method}_${method_sorter}" "python3 scripts/02a-simulation_compute.py --method $method --method-sorter $method_sorter --seeds 1 --max-workers 99";
done
done



# testing

python3 scripts/02a-simulation_compute.py --method uniform_nonsquare --method-sorter random --seeds 100
python3 scripts/02a-simulation_compute.py --method weighted_sampling_rank --method-sorter random --seeds 100
python3 scripts/02a-simulation_compute.py --method weighted_sampling_oracle_rank --method-sorter random --seeds 100
python3 scripts/02a-simulation_compute.py --method uniform --method-sorter random --seeds 100
python3 scripts/02a-simulation_compute.py --method successive_rejects_constant --method-sorter random --seeds 100