#!/bin/bash
# The two graphs on cathedral, each one CPU job on `short`, in the rdnet env.
# A rerun skips the experiments already written. ./run_calib_net.sh <preds dir> [extra eval_calib_net.py args]
set -euo pipefail

PREDS=${1:?the directory of rd_predict.py\'s h5 files}
shift
EXTRA="$*"
CODE=$(cd "$(dirname "$0")" && pwd)
CAT=/home/kocurvik/data/rd/cathedral
FEAT=pairs-features_superpoint_noresize_2048-LG
mkdir -p "$CODE/logs"

for eq in 0 1; do
    if [[ $eq == 1 ]]; then flag=-e; feat=${FEAT}_eq; name=eq; else flag=; feat=$FEAT; name=neq; fi
    sbatch <<EOF
#!/bin/bash
#SBATCH --account p2393-26-3
#SBATCH -p short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -G 0
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=calib_net_$name
#SBATCH -o $CODE/logs/calib_net_$name.%j.out
#SBATCH -e $CODE/logs/calib_net_$name.%j.out
source ~/mambaforge/etc/profile.d/conda.sh
conda activate rdnet
cd $CODE
export PYTHONPATH=$CODE
python eval_calib_net.py --preds $PREDS -nw 64 $flag $EXTRA $feat $CAT
EOF
done
