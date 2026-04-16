# push code to cluster
rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/evaluation-bandit
# get results from cluster
rsync -azP euler:/cluster/work/sachan/vilem/evaluation-bandit/computed/03/ ./computed/03/


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

for data in homo hetero binary likert; do
for estimator in mean additive; do
for method in uniform uniform_nonsquare successive_rejects weighted_sampling_rank weighted_sampling_bolzmann weighted_sampling_epsilongreedy ucb confusion_minimization greedy_oracle_invariant_wtau_pow2 ; do
    sbatch_cpu \
        "synth_${data}_${method}_${estimator}" \
        "python3 scripts/03a-synth_compute.py --data $data --method $method --estimator $estimator --seeds 100 --max-workers 99";
done
done
done