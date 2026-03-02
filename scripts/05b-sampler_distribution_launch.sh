# push code to cluster
rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/evaluation-bandit
# get results from cluster
rsync -azP euler:/cluster/work/sachan/vilem/evaluation-bandit/computed/05/ ./computed/05/


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



sbatch_cpu "sampler_distribution" "python3 scripts/05a-sampler_distribution_compute.py"