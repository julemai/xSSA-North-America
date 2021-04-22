#!/bin/bash

# submit with:
#       sbatch submit-compare_Si_m_to_Markstrom.sh

#SBATCH --mem-per-cpu=5G                          # memory; default unit is megabytes
#SBATCH --time=2-00:00:00
#SBATCH --account=rpp-btolson
#SBATCH --mail-user=juliane.mai@uwaterloo.ca
#SBATCH --mail-type=FAIL
#SBATCH --job-name=markstrom


source /home/julemai/projects/rpp-kshook/julemai/xSSA-North-America/env-3.5/bin/activate

mv sa_for_markstrom_FAST.dat sa_for_markstrom_FAST_save_${SLURM_ARRAY_JOB_ID}.dat

python compare_Si_m_to_Markstrom.py -g test_

# JOBID :: 47102553
