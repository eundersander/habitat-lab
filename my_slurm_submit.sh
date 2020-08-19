#!/bin/bash
#SBATCH --job-name=ddppo
#SBATCH --output=/checkpoint/%u/jobs/job.%j.out
#SBATCH --error=/checkpoint/%u/jobs/job.%j.err
#SBATCH --gpus-per-task 1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 2
#SBATCH --mem-per-cpu=5GB
#SBATCH --partition=learnfair
#SBATCH --time=20:00
#SBATCH --signal=USR1@300
#SBATCH --open-mode=append
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export HABITAT_PROFILING=1
set -x
echo "MASTER_ADDR: $MASTER_ADDR"
srun echo "SLURM_LOCALID: $SLURM_LOCALID"
srun /private/home/eundersander/nsight-systems-2020.3.1/bin/nsys profile --sample=none --trace=nvtx --trace-fork-before-exec=true --output="prof_$SLURM_LOCALID" --export=sqlite python -u -m habitat_baselines.run --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_profiling.yaml --run-type train
