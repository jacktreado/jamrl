#!/bin/bash
#SBATCH --job-name=jamrl-post-backfill
#SBATCH --array=0-116
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --partition=short
#SBATCH --output=/home/treado/data/jamrl/campaigns/sm_N256_P1e-5/logs/backfill_%A_%a.out

# Match this to the --spectra-stride you'll pass to `jamrl analyze` (default 10).
STRIDE=10
CAMP=/home/treado/data/jamrl/campaigns/sm_N256_P1e-5

export OMP_NUM_THREADS=4
export JAMRL_NODE_SCRATCH=/scratch/treado/

conda run -n jamrl python3 - <<EOF
import sys
from jamrl import config, postprocess

camp = "$CAMP"
r    = $SLURM_ARRAY_TASK_ID
stride = $STRIDE

cfg = config.Config.from_yaml(f"{camp}/config.yaml")
# dos_full (full-enthalpy Hessian + mode projection) only for strided rounds;
# all rounds get the backbone eigenvalues + B, G, dz, phi regardless.
dos_full = (r % stride == 0) or (r == 116)
cfg = cfg.replace(dos_full=dos_full)

postprocess.run_postprocess(cfg, camp, r, shard=0, nshards=1)
EOF
